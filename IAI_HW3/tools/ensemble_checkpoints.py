"""Equal-weight softmax ensemble of 1 to 3 checkpoints trained on one fixed split (no training).

Run (needs CUDA; output directory must be new/empty):
    .venv/Scripts/python.exe -m tools.ensemble_checkpoints \
        --checkpoints saved_models/full8100_mlm10/seed42 saved_models/full8100_mlm10/seed43 \
                      saved_models/full8100_mlm10/seed44 \
        --output-dir outputs/full8100_mlm10_ensemble
Each checkpoint is evaluated in FP32 on the split's validation ids and on the benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.training import ROOT, LABELS, load_split, read_rows, write_json
from tools.cv_oof import (BATCH_SIZE, MAX_ENSEMBLE, ensemble_predictions, infer_logits, macro_summary,
                          softmax, write_pred_csv)

__all__ = ["BATCH_SIZE", "check_checkpoint_config", "check_args", "build_report", "run_inference", "main"]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_checkpoint_config(config, split_sha256, dataset_sha256, validation_ids):
    """The checkpoint must come from exactly this split and dataset."""
    if config.get("split_sha256") != split_sha256:
        raise ValueError("Checkpoint split_sha256 differs from the given split")
    if config.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Checkpoint dataset_sha256 differs from the given split")
    if config.get("validation_ids") != validation_ids:
        raise ValueError("Checkpoint validation ids differ from the given split")


def check_args(checkpoints, output_dir):
    if not 1 <= len(checkpoints) <= MAX_ENSEMBLE:
        raise ValueError(f"At most {MAX_ENSEMBLE} models may be ensembled (and at least 1)")
    if len({str(Path(c)) for c in checkpoints}) != len(checkpoints):
        raise ValueError("Checkpoints must be distinct")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory already exists and is not empty")


def run_inference(checkpoint, val_rows, benchmark, split_sha256, dataset_sha256):
    """Load one checkpoint, verify it matches the split, return (config, val_logits, benchmark_logits) in FP32."""
    import gc
    import torch
    from src.training import load_checkpoint, loader
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required")
    model, tokenizer, config = load_checkpoint(Path(checkpoint), torch.device("cuda"))
    check_checkpoint_config(config, split_sha256, dataset_sha256, [r["id"] for r in val_rows])
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise ValueError("Expected FP32 model weights")
    val = infer_logits(model, loader(val_rows, tokenizer, config["max_length"], BATCH_SIZE,
                                     preprocessing=config["preprocessing"]), torch)
    bm = infer_logits(model, loader(benchmark, tokenizer, config["max_length"], BATCH_SIZE, labeled=False,
                                    preprocessing=config["preprocessing"]), torch)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return config, val, bm


def class_report(summary):
    return {k: {"precision": v["precision"], "recall": v["recall"], "f1": v["f1-score"]}
            for k, v in summary["classes"].items()}


def build_report(output_dir, checkpoints, truth, val_logits, bench_ids, bench_logits, split_path):
    """Write per-model and ensemble benchmark CSVs; return the summary dict (with SHA-256 of every CSV)."""
    output_dir = Path(output_dir)
    sub = output_dir / "submissions"
    sub.mkdir(parents=True, exist_ok=True)
    models = []
    for n, (ckpt, val, bm) in enumerate(zip(checkpoints, val_logits, bench_logits)):
        summary = macro_summary(truth, val)
        path = sub / f"model{n}_benchmark_fp32.csv"
        write_pred_csv(path, bench_ids, np.asarray(bm).argmax(axis=1))
        models.append({"checkpoint": str(ckpt), "validation_macro_f1": summary["macro_f1"],
                       "validation_classes": class_report(summary),
                       "benchmark_csv": str(path), "benchmark_csv_sha256": sha256_file(path)})
    idx = list(range(len(checkpoints)))
    # validation: same equal-weight softmax mean used for the benchmark ensemble
    probs = np.mean([softmax(v) for v in val_logits], axis=0)
    ens_summary = macro_summary(truth, probs)
    ens_path = sub / "ensemble_mean_benchmark_fp32.csv"
    write_pred_csv(ens_path, bench_ids, ensemble_predictions(bench_logits, idx))
    report = {"split": str(split_path), "split_sha256": sha256_file(split_path), "labels": list(LABELS),
              "validation_rows": len(truth), "benchmark_rows": len(bench_ids), "models": models,
              "ensemble": {"method": "equal-weight FP32 softmax mean", "n_models": len(models),
                           "validation_macro_f1": ens_summary["macro_f1"],
                           "validation_classes": class_report(ens_summary),
                           "validation_confusion_matrix": ens_summary["confusion_matrix"],
                           "benchmark_csv": str(ens_path), "benchmark_csv_sha256": sha256_file(ens_path)}}
    write_json(output_dir / "summary.json", report)
    return report


def main(argv=None, infer=run_inference):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "outputs/eda/split.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    args = parser.parse_args(argv)
    try:
        check_args(args.checkpoints, args.output_dir)
    except ValueError as error:
        raise SystemExit(str(error))
    _, val_rows, manifest = load_split(args.data_dir, args.split)
    benchmark = read_rows(args.data_dir / "benchmark.csv", labeled=False)
    split_sha, dataset_sha = sha256_file(args.split), manifest["dataset_sha256"]
    truth = [LABELS.index(r["label"]) for r in val_rows]
    val_logits, bench_logits = [], []
    for ckpt in args.checkpoints:  # fail on the first mismatching checkpoint before writing anything
        _, val, bm = infer(ckpt, val_rows, benchmark, split_sha, dataset_sha)
        if np.asarray(val).shape != (len(val_rows), 3) or np.asarray(bm).shape != (len(benchmark), 3):
            raise ValueError("Logit shape mismatch")
        val_logits.append(np.asarray(val))
        bench_logits.append(np.asarray(bm))
        print(f"{ckpt}: {len(val_rows)} validation rows, {len(benchmark)} benchmark rows", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.output_dir, args.checkpoints, truth, val_logits,
                          [r["id"] for r in benchmark], bench_logits, args.split)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
