"""Compare saved HW3 candidates under validation AMP and submission FP32."""

import argparse
import gc
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from src.model_source import file_sha256
from src.training import (ROOT, LABELS, load_checkpoint, load_split, loader,
                          metrics, read_rows, write_json)
from tools.analyze_ensemble import write_csv
from tools.refresh_experiments import read_csv


def compare_outputs(amp_probs, fp32_probs, amp_pred, fp32_pred, ids):
    if amp_probs.shape != fp32_probs.shape or amp_probs.shape != (len(ids), len(LABELS)):
        raise ValueError("Probability shape mismatch")
    if not np.isfinite(amp_probs).all() or not np.isfinite(fp32_probs).all():
        raise ValueError("Non-finite probabilities")
    for values in (amp_probs, fp32_probs):
        np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-6)
    difference = np.abs(amp_probs - fp32_probs)
    changed = np.flatnonzero(amp_pred != fp32_pred)
    return {
        "label_disagreements": len(changed),
        "disagreement_ids": [ids[i] for i in changed],
        "probability_abs_diff_mean": float(difference.mean()),
        "probability_abs_diff_max": float(difference.max()),
        "probability_abs_diff_p99": float(np.quantile(difference, 0.99)),
        "confidence_abs_diff_max": float(np.abs(amp_probs.max(1) - fp32_probs.max(1)).max()),
    }


@torch.inference_mode()
def infer(model, batches, amp, deadline):
    device = torch.device("cuda")
    model.eval()
    # Warm the same first batch for each precision; exclude warmup from timing.
    first = {k: v.to(device) for k, v in next(iter(batches)).items() if k != "labels"}
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            model(**first)
    del first
    torch.cuda.synchronize()
    started = time.perf_counter()
    probabilities, predicted, truth, dtypes, loss_sum = [], [], [], set(), 0.0
    for batch in batches:
        if time.perf_counter() > deadline:
            raise TimeoutError("Precision audit exceeded its inference time budget")
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels", None)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(**batch).logits
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits")
        dtypes.add(str(logits.dtype))
        predicted.extend(logits.argmax(-1).tolist())
        probabilities.append(logits.float().softmax(-1).cpu().numpy())
        if labels is not None:
            loss_sum += F.cross_entropy(logits.float(), labels, reduction="sum").item()
            truth.extend(labels.tolist())
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    expected_dtype = "torch.float16" if amp else "torch.float32"
    if dtypes != {expected_dtype}:
        raise ValueError(f"Unexpected logits dtype: {dtypes}")
    matrix, predictions = np.concatenate(probabilities), np.asarray(predicted)
    sorted_probs = np.sort(matrix, axis=1)
    result = {"seconds": seconds, "rows_per_second": len(predicted) / seconds,
              "logits_dtype": expected_dtype, "rows": len(predicted),
              "mean_confidence": float(matrix.max(1).mean()),
              "min_top2_margin": float((sorted_probs[:, -1] - sorted_probs[:, -2]).min())}
    if truth:
        result.update({"metrics": {"loss": loss_sum / len(truth), **metrics(truth, predicted)},
                       "errors": int(np.count_nonzero(predictions != truth))})
    return matrix, predictions, result


def save_records(destination, rows, values):
    records = []
    for i, row in enumerate(rows):
        record = {"id": row["id"]}
        if "label" in row:
            record["truth"] = row["label"]
        for mode, (probs, pred) in values.items():
            record[f"{mode}_pred"] = LABELS[pred[i]]
            record[f"{mode}_confidence"] = float(probs[i].max())
            ordered = np.sort(probs[i])
            record[f"{mode}_margin"] = float(ordered[-1] - ordered[-2])
            record.update({f"{mode}_p_{label}": float(probs[i, j]) for j, label in enumerate(LABELS)})
        records.append(record)
    write_csv(destination / "probabilities.csv", records)
    write_csv(destination / "disagreements.csv",
              [r for r in records if r["amp_pred"] != r["fp32_pred"]], list(records[0]))
    if "truth" in records[0]:
        write_csv(destination / "errors.csv", [r for r in records if
                  r["amp_pred"] != r["truth"] or r["fp32_pred"] != r["truth"]], list(records[0]))
    else:
        for mode in values:
            write_csv(destination / f"predictions_{mode}.csv",
                      [{"id": r["id"], "pred": r[f"{mode}_pred"]} for r in records])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or (output.exists() and any(output.iterdir())):
        raise ValueError("Use a new or empty output directory inside HW3")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required to compare FP16 AMP with FP32")
    split_path = ROOT / "outputs/eda/split.json"
    _, validation, split = load_split(ROOT / "dataset", split_path)
    benchmark = read_rows(ROOT / "dataset/benchmark.csv", labeled=False)
    if len(validation) != 900 or len(benchmark) != 1000:
        raise ValueError("Expected 900 validation and 1,000 benchmark rows")
    snapshot = json.loads((ROOT / "experiments/snapshot.json").read_text(encoding="utf-8"))
    sources = ["dataset/dataset.csv", "dataset/benchmark.csv", "outputs/eda/split.json",
               "main.py", "src/training.py", "src/eda.py", "src/preprocessing.py", "src/model_source.py",
               "tools/audit_precision.py", "tools/analyze_ensemble.py", "tools/refresh_experiments.py",
               "requirements/lock.txt"]
    summary = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "labels": list(LABELS), "training_runs": 0, "inference_pass_budget": 8,
               "inference_time_limit_seconds": 1800, "warmup_batches_per_pass": 3,
               "timing_scope": "One warmed full loop per mode; includes transfers, softmax, and validation loss; excludes loading/tokenization/writes. Not a calibrated speed benchmark.",
               "source_sha256": {p: file_sha256(ROOT / p) for p in sources},
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "transformers": transformers.__version__, "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(),
                               "float32_matmul_precision": torch.get_float32_matmul_precision(),
                               "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
               "validation_ids": [r["id"] for r in validation], "benchmark_ids": [r["id"] for r in benchmark],
               "runs": {}, "completed": False}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    deadline = time.perf_counter() + 1800
    for name in ("weighted_sqrt", "weighted_lr3e5"):
        checkpoint = f"outputs/{name}/checkpoint"
        expected_hashes = snapshot["checkpoints"][checkpoint]["files_sha256"]
        hashes = {p: file_sha256(ROOT / checkpoint / p) for p in expected_hashes}
        if hashes != expected_hashes:
            raise ValueError(f"Checkpoint differs from existing snapshot: {name}")
        model, tokenizer, config = load_checkpoint(ROOT / checkpoint, torch.device("cuda"))
        if (config["validation_ids"] != summary["validation_ids"] or config["train_ids"] != split["train_ids"]
                or config["dataset_sha256"] != summary["source_sha256"]["dataset/dataset.csv"]
                or config["split_sha256"] != summary["source_sha256"]["outputs/eda/split.json"]
                or config["smoke"] or not config["amp"] or config["batch_size"] != 2):
            raise ValueError("Checkpoint does not match the planned full validation comparison")
        if {p.dtype for p in model.parameters()} != {torch.float32}:
            raise ValueError("Expected FP32 model weights")
        saved_path = f"outputs/{name}/predictions.csv"
        saved = read_csv(saved_path, ["id", "pred"])
        if [r["id"] for r in saved] != summary["benchmark_ids"] or any(r["pred"] not in LABELS for r in saved):
            raise ValueError("Saved CSV IDs/labels differ from benchmark")
        run = {"checkpoint": checkpoint, "checkpoint_files_sha256": hashes,
               "saved_predictions": saved_path, "saved_predictions_sha256": file_sha256(ROOT / saved_path),
               "batch_size": 2, "max_length": config["max_length"], "preprocessing": config["preprocessing"]}
        summary["runs"][name] = run
        for split_name, rows in (("validation", validation), ("benchmark", benchmark)):
            batches = loader(rows, tokenizer, config["max_length"], 2, labeled=split_name == "validation",
                             preprocessing=config["preprocessing"])
            values, results = {}, {}
            for mode in ("amp", "fp32"):
                probs, pred, result = infer(model, batches, mode == "amp", deadline)
                values[mode], results[mode] = (probs, pred), result
                print(f"{name} {split_name} {mode}: {result['rows']} rows, {result['seconds']:.2f}s", flush=True)
            results["comparison"] = compare_outputs(values["amp"][0], values["fp32"][0],
                                                     values["amp"][1], values["fp32"][1], [r["id"] for r in rows])
            if split_name == "validation":
                current, historical = results["amp"]["metrics"], config["validation"]
                results["amp_reproduces_saved_validation"] = (
                    current["confusion_matrix"] == historical["confusion_matrix"]
                    and math.isclose(current["loss"], historical["loss"], rel_tol=1e-5, abs_tol=1e-6))
                results["comparison"]["fp32_minus_amp_macro_f1"] = (
                    results["fp32"]["metrics"]["macro_f1"] - current["macro_f1"])
            else:
                saved_labels = np.array([LABELS.index(r["pred"]) for r in saved])
                results["differences_from_saved_csv"] = {
                    mode: int(np.count_nonzero(pred != saved_labels)) for mode, (_, pred) in values.items()}
            destination = output / name / split_name
            destination.mkdir(parents=True)
            save_records(destination, rows, values)
            run[split_name] = results
            write_json(output / "summary.json", summary)
        del model, tokenizer, batches
        gc.collect()
        torch.cuda.empty_cache()
    summary["historical_reproduction_passed"] = all(
        r["validation"]["amp_reproduces_saved_validation"] and r["benchmark"]["differences_from_saved_csv"]["fp32"] == 0
        for r in summary["runs"].values())
    summary["all_precision_labels_identical"] = all(
        r[s]["comparison"]["label_disagreements"] == 0 for r in summary["runs"].values() for s in ("validation", "benchmark"))
    summary["artifacts_sha256"] = {p.relative_to(output).as_posix(): file_sha256(p) for p in sorted(output.rglob("*.csv"))}
    summary["completed"] = True
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("completed", "historical_reproduction_passed", "all_precision_labels_identical")}), flush=True)
    if not summary["historical_reproduction_passed"]:
        raise ValueError("Historical reproduction failed; inspect preserved audit results before proceeding")


if __name__ == "__main__":
    main()
