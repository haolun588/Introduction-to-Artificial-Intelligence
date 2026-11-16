"""Out-of-fold (OOF) evaluation and fixed-fold ensemble for grouped k-fold checkpoints (no training).

Run (needs CUDA; output directory must be new/empty;
the default ensemble CSV is written to <output-dir>/submissions/cv5_folds012_mean.csv unless --submission is given):
    .venv/Scripts/python.exe -m tools.cv_oof --fold-dir outputs/cv5_seed42 \
        --checkpoints saved_models/cv5_sqrt/fold0 ... saved_models/cv5_sqrt/fold4 \
        --output-dir outputs/cv5_sqrt_oof
"""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from src.training import ROOT, LABELS, load_split, metrics, read_rows, write_json

BATCH_SIZE = 2  # same as the FP32 precision audit
MAX_ENSEMBLE = 3
DEFAULT_ENSEMBLE_FOLDS = (0, 1, 2)  # fixed in advance, not chosen by score


def check_fold_split(config, fold_manifest):
    """The checkpoint must have been trained on exactly this fold's train/validation ids."""
    if (config.get("train_ids") != fold_manifest["train_ids"]
            or config.get("validation_ids") != fold_manifest["validation_ids"]):
        raise ValueError("Checkpoint split differs from the fold JSON")


def assemble_oof(fold_results, all_ids):
    """fold_results: list of (ids, logits). Returns logits ordered as all_ids; each id exactly once."""
    seen = {}
    for ids, logits in fold_results:
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != (len(ids), len(LABELS)):
            raise ValueError("Logit shape mismatch")
        for i, row in zip(ids, logits):
            if i in seen:
                raise ValueError(f"Duplicate OOF id: {i}")
            seen[i] = row
    missing = set(all_ids) - set(seen)
    extra = set(seen) - set(all_ids)
    if missing or extra:
        raise ValueError(f"OOF ids mismatch: missing {len(missing)}, unexpected {len(extra)}")
    return np.stack([seen[i] for i in all_ids])


def macro_summary(truth, logits):
    return metrics(list(map(int, truth)), np.asarray(logits).argmax(axis=1).tolist())


def fold_statistics(scores):
    return {"per_fold": list(scores), "mean": float(statistics.mean(scores)),
            "sample_std": float(statistics.stdev(scores)) if len(scores) > 1 else None}


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ensemble_predictions(bench_logits, folds):
    """Equal-weight softmax mean over the chosen folds (at most 3); returns argmax indices."""
    folds = list(folds)
    if not 1 <= len(folds) <= MAX_ENSEMBLE or len(set(folds)) != len(folds):
        raise ValueError(f"Ensemble must use 1 to {MAX_ENSEMBLE} distinct folds")
    if any(f not in range(len(bench_logits)) for f in folds):
        raise ValueError("Ensemble fold index out of range")
    return np.mean([softmax(bench_logits[f]) for f in folds], axis=0).argmax(axis=1)


def write_pred_csv(path, ids, pred):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "pred"])
        writer.writerows([i, LABELS[p]] for i, p in zip(ids, pred))


def write_oof_csv(path, ids, truth, logits):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label", "pred"] + [f"logit_{k}" for k in LABELS])
        for i, t, row in zip(ids, truth, logits):
            writer.writerow([i, LABELS[t], LABELS[int(row.argmax())]] + [repr(float(x)) for x in row])


def infer_logits(model, batches, torch):
    device = next(model.parameters()).device
    out = []
    with torch.inference_mode():
        for batch in batches:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits  # FP32, no autocast
            if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                raise ValueError("Expected finite FP32 logits")
            out.append(logits.cpu().numpy())
    return np.concatenate(out)


def run_inference(output, fold_dir, manifest, checkpoints, dataset_rows, benchmark):
    import gc
    import torch
    from src.training import load_checkpoint, loader
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required")
    by_id = {r["id"]: r for r in dataset_rows}
    results, bench = [], []
    for entry, checkpoint in zip(manifest["folds"], checkpoints):
        i = entry["fold"]
        fold = json.loads((fold_dir / entry["file"]).read_text(encoding="utf-8"))
        model, tokenizer, config = load_checkpoint(checkpoint, torch.device("cuda"))
        check_fold_split(config, fold)
        if {p.dtype for p in model.parameters()} != {torch.float32}:
            raise ValueError("Expected FP32 model weights")
        rows = [by_id[x] for x in fold["validation_ids"]]
        val = infer_logits(model, loader(rows, tokenizer, config["max_length"], BATCH_SIZE,
                                         preprocessing=config["preprocessing"]), torch)
        bm = infer_logits(model, loader(benchmark, tokenizer, config["max_length"], BATCH_SIZE, labeled=False,
                                        preprocessing=config["preprocessing"]), torch)
        if val.shape != (len(rows), 3) or bm.shape != (len(benchmark), 3):
            raise ValueError("Logit shape mismatch")
        np.savez(output / f"logits_fold{i}.npz", ids=np.array(fold["validation_ids"]), logits=val)
        np.savez(output / f"logits_benchmark_fold{i}.npz", ids=np.array([r["id"] for r in benchmark]), logits=bm)
        results.append((fold["validation_ids"], val))
        bench.append(bm)
        print(f"fold {i}: {len(rows)} validation rows, {len(benchmark)} benchmark rows", flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    return results, bench


def evaluate(output, dataset_rows, results, manifest, reference_split):
    all_ids = [r["id"] for r in dataset_rows]
    truth = np.array([LABELS.index(r["label"]) for r in dataset_rows])
    oof = assemble_oof(results, all_ids)
    overall = macro_summary(truth, oof)
    index = {i: n for n, i in enumerate(all_ids)}
    fold_scores = [macro_summary([truth[index[i]] for i in ids], logits)["macro_f1"] for ids, logits in results]
    report = {"k": manifest["k"], "seed": manifest["seed"], "dataset_sha256": manifest["dataset_sha256"],
              "rows": len(all_ids), "labels": list(LABELS),
              "oof": {"macro_f1": overall["macro_f1"],
                      "classes": {k: {"precision": v["precision"], "recall": v["recall"], "f1": v["f1-score"]}
                                  for k, v in overall["classes"].items()},
                      "confusion_matrix": overall["confusion_matrix"]},
              "folds": fold_statistics(fold_scores)}
    if reference_split is not None:
        ref_ids = reference_split["validation_ids"]
        pos = [index[i] for i in ref_ids]
        sub = macro_summary(truth[pos], oof[pos])
        report["original_900_validation_ids_in_oof"] = {"rows": len(ref_ids), "macro_f1": sub["macro_f1"]}
    write_oof_csv(output / "oof_predictions.csv", all_ids, truth, oof)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--reference-split", type=Path, default=ROOT / "outputs/eda/split.json")
    parser.add_argument("--ensemble-folds", type=int, nargs="+", default=list(DEFAULT_ENSEMBLE_FOLDS))
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if len(args.ensemble_folds) > MAX_ENSEMBLE:
        raise SystemExit(f"At most {MAX_ENSEMBLE} models may be ensembled")
    manifest = json.loads((args.fold_dir / "folds_manifest.json").read_text(encoding="utf-8"))
    if len(args.checkpoints) != manifest["k"]:
        raise SystemExit(f"Expected {manifest['k']} checkpoints, got {len(args.checkpoints)}")
    for entry in manifest["folds"]:  # fold files must be unchanged since creation
        if hashlib.sha256((args.fold_dir / entry["file"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise SystemExit(f"Fold file hash differs from manifest: {entry['file']}")
    name = "cv%d_folds%s_mean.csv" % (manifest["k"], "".join(map(str, args.ensemble_folds)))
    submission = args.submission or args.output_dir / "submissions" / name
    if args.submission is not None and submission.exists():  # default path lives in the new output dir
        raise SystemExit(f"Submission file already exists: {submission}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("Output directory already exists and is not empty")
    for entry in manifest["folds"]:  # also validates each fold with the training loader
        load_split(args.data_dir, args.fold_dir / entry["file"])
    dataset_rows = read_rows(args.data_dir / "dataset.csv")
    benchmark = read_rows(args.data_dir / "benchmark.csv", labeled=False)
    reference = (json.loads(args.reference_split.read_text(encoding="utf-8"))
                 if args.reference_split.exists() else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, bench = run_inference(args.output_dir, args.fold_dir, manifest, args.checkpoints,
                                   dataset_rows, benchmark)
    report = evaluate(args.output_dir, dataset_rows, results, manifest, reference)
    pred = ensemble_predictions(bench, args.ensemble_folds)
    submission.parent.mkdir(parents=True, exist_ok=True)
    write_pred_csv(submission, [r["id"] for r in benchmark], pred)
    report["ensemble"] = {"folds": args.ensemble_folds, "method": "equal-weight FP32 softmax mean",
                          "file": str(submission), "sha256": hashlib.sha256(submission.read_bytes()).hexdigest()}
    write_json(args.output_dir / "oof_metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
