"""Offline recombination of per-fold benchmark logits (no training, no inference, no GPU).

Same method as tools.cv_oof.ensemble_predictions (FP32 logits -> softmax -> equal-weight mean -> argmax),
but without the 3-fold limit, and writing the CSV with tools.cv_oof.write_pred_csv:
    .venv/Scripts/python.exe -m tools.recombine_folds --oof-dir outputs/cv5_sqrt_ep5_mlm10_oof \
        --folds 0 1 2 --output outputs/cv5_sqrt_ep5_mlm10_recombine/check_folds012.csv
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from src.training import LABELS
from tools.cv_oof import softmax, write_pred_csv

NUM_FOLDS = 5


def load_fold_logits(oof_dir, folds):
    """Returns (ids, {fold: logits}); all folds must share identical ids in identical order."""
    ids, out = None, {}
    for f in folds:
        path = Path(oof_dir) / f"logits_benchmark_fold{f}.npz"
        if not path.is_file():
            raise ValueError(f"Missing logits file: {path}")
        with np.load(path, allow_pickle=False) as data:
            fold_ids = [str(x) for x in data["ids"]]
            logits = np.asarray(data["logits"])
        if logits.shape != (len(fold_ids), len(LABELS)):
            raise ValueError(f"Logit shape mismatch in fold {f}: {logits.shape}")
        if not np.isfinite(logits).all():
            raise ValueError(f"Non-finite logits in fold {f}")
        if ids is None:
            ids = fold_ids
        elif fold_ids != ids:
            raise ValueError(f"Benchmark ids of fold {f} differ from fold {folds[0]}")
        out[f] = logits
    return ids, out


def check_folds(folds):
    folds = list(folds)
    if not 1 <= len(folds) <= NUM_FOLDS or len(set(folds)) != len(folds):
        raise ValueError(f"Use 1 to {NUM_FOLDS} distinct folds")
    if any(f not in range(NUM_FOLDS) for f in folds):
        raise ValueError(f"Fold index must be in 0..{NUM_FOLDS - 1}")
    return folds


def mean_softmax_predictions(fold_logits, folds):
    """Equal-weight softmax mean over the chosen folds; returns argmax indices (same as cv_oof)."""
    folds = check_folds(folds)
    return np.mean([softmax(fold_logits[f]) for f in folds], axis=0).argmax(axis=1)


def read_pred_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if rows[0] != ["id", "pred"]:
        raise ValueError("Reference CSV header must be id,pred")
    return [r[0] for r in rows[1:]], [r[1] for r in rows[1:]]


def class_counts(pred):
    c = Counter(int(p) for p in pred)
    return {LABELS[k]: c.get(k, 0) for k in range(len(LABELS))}


def disagreement(pred_a, pred_b):
    """Number of differing predictions and (a->b) transition counts."""
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    if pred_a.shape != pred_b.shape:
        raise ValueError("Prediction length mismatch")
    diff = pred_a != pred_b
    trans = Counter((LABELS[int(a)], LABELS[int(b)]) for a, b in zip(pred_a[diff], pred_b[diff]))
    return int(diff.sum()), {f"{a}->{b}": n for (a, b), n in sorted(trans.items())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--oof-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default=None, help="reference id,pred CSV for disagreement counts")
    args = parser.parse_args(argv)

    folds = check_folds(args.folds)
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    ids, fold_logits = load_fold_logits(args.oof_dir, folds)
    pred = mean_softmax_predictions(fold_logits, folds)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_pred_csv(output, ids, pred)
    print(f"wrote {output} ({len(ids)} rows, folds {folds})")
    print("prediction counts:", class_counts(pred))
    if args.reference:
        ref_ids, ref_pred = read_pred_csv(args.reference)
        if ref_ids != ids:
            raise ValueError("Reference ids differ from logits ids")
        ref = np.array([LABELS.index(p) for p in ref_pred])
        n, trans = disagreement(ref, pred)
        print(f"disagreements vs reference: {n}")
        print("transitions (ref->new):", trans)


if __name__ == "__main__":
    main()
