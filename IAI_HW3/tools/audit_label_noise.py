"""Label-noise audit sampler over OOF predictions (no training, no GPU).

Run:
    .venv/Scripts/python.exe -m tools.audit_label_noise --oof outputs/cv5_sqrt_ep5_oof/oof_predictions.csv \
        --output-dir outputs/label_noise_audit_20261116
Writes review_blind.csv (text only), review_key.csv, candidates_high_conf.csv and summary.json.
The output directory must not exist or must be empty.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.training import ROOT, LABELS, read_rows, write_json

DEFAULT_SEED = 20261116
THRESHOLDS = (0.70, 0.80, 0.90, 0.95)
HIGH_CONF = 0.90
A_LIMIT, B_LIMIT, C_LIMIT = 40, 20, 20
NEUTRAL = "neutral"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def read_oof(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_inputs(oof_rows, data_rows):
    oof_ids = [r["id"] for r in oof_rows]
    if len(set(oof_ids)) != len(oof_ids):
        raise ValueError("Duplicate ids in OOF predictions")
    data = {r["id"]: r for r in data_rows}
    if set(oof_ids) != set(data):
        raise ValueError("OOF ids and dataset ids differ")
    for r in oof_rows:
        if r["label"] != data[r["id"]]["label"]:
            raise ValueError(f"Label mismatch for id {r['id']}")
    return data


def combo_counts(mask, label, pred):
    return {f"{a}->{b}": int((mask & (label == LABELS.index(a)) & (pred == LABELS.index(b))).sum())
            for a in LABELS for b in LABELS if a != b}


def analyze(oof_rows, data_rows, seed=DEFAULT_SEED):
    """Pure computation: returns (stats, key_rows, blind_rows, candidates)."""
    data = check_inputs(oof_rows, data_rows)
    ids = [r["id"] for r in oof_rows]
    label = np.array([LABELS.index(r["label"]) for r in oof_rows])
    pred = np.array([LABELS.index(r["pred"]) for r in oof_rows])
    probs = softmax([[float(r[f"logit_{l}"]) for l in LABELS] for r in oof_rows])
    rows_idx = np.arange(len(ids))
    p_pred, p_label = probs[rows_idx, pred], probs[rows_idx, label]
    disagree = pred != label
    high = disagree & (p_pred >= HIGH_CONF)

    thresholds = {}
    for t in THRESHOLDS:
        sel = disagree & (p_pred >= t)
        thresholds[f"{t:.2f}"] = {"total": int(sel.sum()), "by_label_pred": combo_counts(sel, label, pred)}
    stats = {"rows": len(ids), "class_counts": {l: int((label == i).sum()) for i, l in enumerate(LABELS)},
             "disagreement": {"total": int(disagree.sum()), "by_label_pred": combo_counts(disagree, label, pred)},
             "high_conf_threshold": HIGH_CONF, "high_conf_count": int(high.sum()),
             "pred_prob_thresholds": thresholds}

    rng = np.random.default_rng(seed)
    neutral_idx = LABELS.index(NEUTRAL)
    # Stable ordering: p_pred descending, ties by OOF row order.
    a_pool = np.flatnonzero(high & (label == neutral_idx))
    a_pick = a_pool[np.argsort(-p_pred[a_pool], kind="stable")][:A_LIMIT]
    b_pool = np.flatnonzero(high & (label != neutral_idx))
    b_pick = b_pool if len(b_pool) <= B_LIMIT else np.sort(rng.choice(b_pool, B_LIMIT, replace=False))
    c_pool = np.flatnonzero(~disagree)
    c_pick = c_pool if len(c_pool) <= C_LIMIT else np.sort(rng.choice(c_pool, C_LIMIT, replace=False))
    picked = [(int(i), "A") for i in a_pick] + [(int(i), "B") for i in b_pick] + [(int(i), "C") for i in c_pick]
    key_rows, blind_rows = [], []
    for n, k in enumerate(rng.permutation(len(picked)), start=1):
        i, group = picked[int(k)]
        review_id = f"R{n:03d}"
        text = data[ids[i]]["text"]
        key_rows.append({"review_id": review_id, "id": ids[i], "group": group, "label": LABELS[label[i]],
                         "pred": LABELS[pred[i]], "p_pred": float(p_pred[i]), "p_label": float(p_label[i]),
                         "text": text})
        blind_rows.append({"review_id": review_id, "text": text})
    high_idx = np.flatnonzero(high)
    high_idx = high_idx[np.argsort(-p_pred[high_idx], kind="stable")]
    candidates = [{"id": ids[i], "label": LABELS[label[i]], "pred": LABELS[pred[i]], "p_pred": float(p_pred[i]),
                   "p_label": float(p_label[i]), "text": data[ids[i]]["text"]} for i in high_idx]
    stats["group_sizes"] = {"A": len(a_pick), "B": len(b_pick), "C": len(c_pick)}
    return stats, key_rows, blind_rows, candidates


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def check_output_dir(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SystemExit(f"Output directory exists and is not empty: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=ROOT / "outputs/cv5_sqrt_ep5_oof/oof_predictions.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--fold-dir", type=Path, default=ROOT / "outputs/cv5_seed42")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    check_output_dir(args.output_dir)
    dataset_path = args.data_dir / "dataset.csv"
    data_rows = read_rows(dataset_path)
    manifest_path = args.fold_dir / "folds_manifest.json"
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text(encoding="utf-8")).get("dataset_sha256")
        if expected and expected != sha256_file(dataset_path):
            raise SystemExit("dataset.csv sha256 differs from folds_manifest dataset_sha256")
    stats, key_rows, blind_rows, candidates = analyze(read_oof(args.oof), data_rows, args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "review_blind.csv", ["review_id", "text"], blind_rows)
    write_csv(out / "review_key.csv",
              ["review_id", "id", "group", "label", "pred", "p_pred", "p_label", "text"], key_rows)
    write_csv(out / "candidates_high_conf.csv", ["id", "label", "pred", "p_pred", "p_label", "text"], candidates)
    summary = dict(stats)
    summary.update({"seed": args.seed,
                    "inputs": {"oof": {"path": str(args.oof), "sha256": sha256_file(args.oof)},
                               "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)}},
                    "outputs_sha256": {n: sha256_file(out / n) for n in
                                       ("review_blind.csv", "review_key.csv", "candidates_high_conf.csv")}})
    write_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("rows", "disagreement", "high_conf_count", "group_sizes")}, indent=2))


if __name__ == "__main__":
    main()
