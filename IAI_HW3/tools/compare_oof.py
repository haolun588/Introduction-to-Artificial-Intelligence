"""Paired comparison of two OOF evaluations on the same grouped k-fold split (no training).

Run:
    .venv/Scripts/python.exe -m tools.compare_oof --a outputs/cv5_sqrt_oof --b outputs/cv5_quarter_oof \
        --output outputs/cv5_compare_sqrt_vs_quarter.json
Delta is always B - A. The output file must not exist.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.training import ROOT, LABELS, metrics, write_json

N_CLASSES = len(LABELS)
DEFAULT_SEED = 20261116
DEFAULT_RESAMPLES = 2000
FOLD_SCORE_TOLERANCE = 1e-9


def macro_f1_from_confusion(conf):
    """Macro-F1 from a (3, 3) confusion matrix (rows = truth); an empty class scores 0, like zero_division=0."""
    conf = np.asarray(conf, dtype=np.float64)
    tp = np.diag(conf)
    denominator = conf.sum(axis=0) + conf.sum(axis=1)  # 2tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return float(f1.mean())


def confusion(truth, pred):
    return np.bincount(np.asarray(truth) * N_CLASSES + np.asarray(pred),
                       minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)


def macro_f1(truth, pred):
    return macro_f1_from_confusion(confusion(truth, pred))


def read_oof_csv(path):
    ids, truth, pred = [], [], []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            ids.append(row["id"])
            truth.append(LABELS.index(row["label"]))
            pred.append(LABELS.index(row["pred"]))
    return ids, np.array(truth), np.array(pred)


def load_side(directory):
    directory = Path(directory)
    metrics_json = json.loads((directory / "oof_metrics.json").read_text(encoding="utf-8"))
    ids, truth, pred = read_oof_csv(directory / "oof_predictions.csv")
    return {"metrics": metrics_json, "ids": ids, "truth": truth, "pred": pred}


def check_compatible(a, b):
    for key in ("dataset_sha256", "k", "seed"):
        if a["metrics"][key] != b["metrics"][key]:
            raise ValueError(f"oof_metrics {key} differs: {a['metrics'][key]!r} vs {b['metrics'][key]!r}")
    if a["ids"] != b["ids"]:
        raise ValueError("OOF id order differs")
    if not np.array_equal(a["truth"], b["truth"]):
        raise ValueError("OOF labels differ")


def load_fold_ids(fold_dir):
    """Validation ids per fold; fold files must match the sha256 recorded in the manifest."""
    fold_dir = Path(fold_dir)
    manifest = json.loads((fold_dir / "folds_manifest.json").read_text(encoding="utf-8"))
    folds = []
    for entry in manifest["folds"]:
        data = (fold_dir / entry["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Fold file hash differs from manifest: {entry['file']}")
        folds.append(json.loads(data.decode("utf-8"))["validation_ids"])
    return folds


def fold_positions(ids, fold_ids):
    index = {i: n for n, i in enumerate(ids)}
    positions = [np.array([index[i] for i in fold]) for fold in fold_ids]
    if sorted(np.concatenate(positions).tolist()) != list(range(len(ids))):
        raise ValueError("Fold validation ids are not a partition of the OOF ids")
    return positions


def fold_scores(truth, pred, positions):
    return [macro_f1(truth[p], pred[p]) for p in positions]


def check_fold_consistency(side, scores, label):
    """oof_metrics records per-fold scores; recomputing them on the fold split ties the OOF file to that split."""
    recorded = side["metrics"]["folds"]["per_fold"]
    if len(recorded) != len(scores) or any(abs(r - s) > FOLD_SCORE_TOLERANCE for r, s in zip(recorded, scores)):
        raise ValueError(f"Side {label}: per-fold scores in oof_metrics do not match the fold split")


def paired_bootstrap(truth, pred_a, pred_b, seed=DEFAULT_SEED, resamples=DEFAULT_RESAMPLES):
    """Resample rows with replacement; both sides are scored on the same resample. Delta = B - A."""
    rng = np.random.default_rng(seed)
    n = len(truth)
    deltas = np.empty(resamples)
    for r in range(resamples):
        idx = rng.integers(0, n, n)
        t = truth[idx]
        deltas[r] = macro_f1(t, pred_b[idx]) - macro_f1(t, pred_a[idx])
    return {"seed": seed, "resamples": resamples,
            "delta_p2_5": float(np.percentile(deltas, 2.5)), "delta_p97_5": float(np.percentile(deltas, 97.5)),
            "delta_mean": float(deltas.mean()), "fraction_delta_le_0": float((deltas <= 0).mean())}


def side_summary(side):
    overall = metrics(side["truth"].tolist(), side["pred"].tolist())
    return {"oof_macro_f1": overall["macro_f1"],
            "class_f1": {k: v["f1-score"] for k, v in overall["classes"].items()}}


def compare(a, b, fold_ids, seed=DEFAULT_SEED, resamples=DEFAULT_RESAMPLES):
    check_compatible(a, b)
    positions = fold_positions(a["ids"], fold_ids)
    scores_a = fold_scores(a["truth"], a["pred"], positions)
    scores_b = fold_scores(b["truth"], b["pred"], positions)
    check_fold_consistency(a, scores_a, "A")
    check_fold_consistency(b, scores_b, "B")
    sum_a, sum_b = side_summary(a), side_summary(b)
    differ = a["pred"] != b["pred"]
    truth = a["truth"]
    per_fold_delta = [sb - sa for sa, sb in zip(scores_a, scores_b)]
    start = time.perf_counter()
    boot = paired_bootstrap(truth, a["pred"], b["pred"], seed, resamples)
    boot["seconds"] = time.perf_counter() - start
    return {"rows": len(truth), "k": a["metrics"]["k"], "seed": a["metrics"]["seed"],
            "dataset_sha256": a["metrics"]["dataset_sha256"],
            "split_check": "fold files verified against folds_manifest sha256; per-fold Macro-F1 recomputed from "
                           "the OOF predictions equals oof_metrics on both sides",
            "A": sum_a, "B": sum_b,
            "delta_oof_macro_f1": sum_b["oof_macro_f1"] - sum_a["oof_macro_f1"],
            "delta_class_f1": {k: sum_b["class_f1"][k] - sum_a["class_f1"][k] for k in LABELS},
            "per_fold": {"A": scores_a, "B": scores_b, "delta": per_fold_delta,
                         "b_wins": int(sum(d > 0 for d in per_fold_delta)),
                         "ties": int(sum(d == 0 for d in per_fold_delta))},
            "paired_bootstrap": boot,
            "disagreement": {"rows": int(differ.sum()),
                             "a_correct": int((differ & (a["pred"] == truth)).sum()),
                             "b_correct": int((differ & (b["pred"] == truth)).sum())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="baseline OOF directory")
    parser.add_argument("--b", type=Path, required=True, help="candidate OOF directory (Delta = B - A)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, default=ROOT / "outputs/cv5_seed42")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    fold_ids = load_fold_ids(args.fold_dir)
    report = compare(load_side(args.a), load_side(args.b), fold_ids, args.seed, args.resamples)
    report["a_dir"], report["b_dir"] = str(args.a), str(args.b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
