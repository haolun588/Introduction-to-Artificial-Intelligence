"""Post-hoc logit adjustment for saved HW3 checkpoints (no training).

adjusted = logits - tau * log(pi), pi = class proportions of the 8,100-row training split.
tau > 0 favours minority classes, tau < 0 favours the majority class.

Run (needs CUDA, output directory must be new/empty):
    .venv/Scripts/python.exe -m tools.logit_adjust
Re-verify from the saved logits without inference:
    .venv/Scripts/python.exe -m tools.logit_adjust --check
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.training import ROOT, LABELS, load_split, metrics, read_rows, write_json

OUTPUT_REL = "outputs/logit_adjust_20261116"
RUNS = ("weighted_sqrt", "weighted_quarter")
REFERENCE_RUN = "weighted_quarter"  # current best Kaggle CSV
# Fixed in advance: -1.0 .. 1.0 step 0.1 (21 points).
TAU_GRID = (-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
FLAT_TOLERANCE = 0.002  # 0.2 percentage points of validation Macro-F1
TIE_EPS = 1e-12
BATCH_SIZE = 2  # same as the FP32 precision audit


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def class_prior(labels):
    """Class proportions in LABELS order from a list of label strings."""
    counts = np.array([sum(1 for x in labels if x == name) for name in LABELS], dtype=np.float64)
    if counts.min() <= 0:
        raise ValueError("Every class must appear in the training split")
    return counts, counts / counts.sum()


def adjust_logits(logits, prior, tau):
    return np.asarray(logits, dtype=np.float64) - tau * np.log(np.asarray(prior, dtype=np.float64))


def predict_adjusted(logits, prior, tau):
    return adjust_logits(logits, prior, tau).argmax(axis=1)


def select_tau(taus, scores):
    """Highest score; ties (within TIE_EPS) -> smallest |tau|, then smaller tau."""
    best = max(scores)
    tied = [t for t, s in zip(taus, scores) if s >= best - TIE_EPS]
    return min(tied, key=lambda t: (abs(t), t))


def flat_taus(taus, scores, tolerance=FLAT_TOLERANCE):
    best = max(scores)
    return [t for t, s in zip(taus, scores) if s >= best - tolerance - TIE_EPS]


def read_pred_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "pred"]:
            raise ValueError(f"Unexpected CSV header: {path}")
        rows = list(reader)
    if any(r["pred"] not in LABELS for r in rows):
        raise ValueError(f"Illegal label in {path}")
    return [r["id"] for r in rows], np.array([LABELS.index(r["pred"]) for r in rows])


def write_pred_csv(path, ids, pred):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "pred"])
        writer.writerows([i, LABELS[p]] for i, p in zip(ids, pred))


def check_pred_csv(path, expected_ids):
    ids, _ = read_pred_csv(path)
    if ids != list(expected_ids) or len(ids) != 1000:
        raise ValueError(f"Submission IDs differ from benchmark order: {path}")


def evaluate_run(val_logits, val_truth, bench_logits, prior, original_pred, reference_pred):
    """Per-tau metrics for one run. truth/pred are integer index arrays."""
    rows = []
    for tau in TAU_GRID:
        m = metrics(val_truth.tolist(), predict_adjusted(val_logits, prior, tau).tolist())
        pred = predict_adjusted(bench_logits, prior, tau)
        rows.append({
            "tau": tau, "val_macro_f1": m["macro_f1"],
            "classes": {k: {"precision": m["classes"][k]["precision"], "recall": m["classes"][k]["recall"],
                            "f1": m["classes"][k]["f1-score"]} for k in LABELS},
            "val_confusion_matrix": m["confusion_matrix"],
            "benchmark_counts": {k: int(np.count_nonzero(pred == i)) for i, k in enumerate(LABELS)},
            "benchmark_diff_vs_own_csv": int(np.count_nonzero(pred != original_pred)),
            "benchmark_diff_vs_weighted_quarter_csv": int(np.count_nonzero(pred != reference_pred)),
        })
    return rows


def table_rows(results):
    out = []
    for run, rows in results.items():
        for r in rows:
            rec = {"run": run, "tau": r["tau"], "val_macro_f1": r["val_macro_f1"]}
            for k in LABELS:
                for metric in ("precision", "recall", "f1"):
                    rec[f"val_{k}_{metric}"] = r["classes"][k][metric]
            rec["val_confusion_matrix"] = json.dumps(r["val_confusion_matrix"])
            for k in LABELS:
                rec[f"bench_pred_{k}"] = r["benchmark_counts"][k]
            rec["bench_diff_vs_own_csv"] = r["benchmark_diff_vs_own_csv"]
            rec["bench_diff_vs_weighted_quarter_csv"] = r["benchmark_diff_vs_weighted_quarter_csv"]
            out.append(rec)
    return out


def write_table(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_logits(directory):
    data = {}
    for split in ("val", "benchmark"):
        with np.load(directory / f"logits_{split}.npz", allow_pickle=False) as z:
            data[split] = {"id": [str(x) for x in z["id"]], "logits": z["logits"].astype(np.float64),
                           "labels": [str(x) for x in z["labels"]],
                           "truth": z["truth"].copy() if "truth" in z.files else None}
    return data


def build(output, val_ids, bench_ids, val_truth, prior, counts, write):
    """Deterministic computation from saved logits; writes files only when write=True.

    Returns (summary, table, submissions: {name: (ids, pred)}) and the acceptance record.
    """
    existing = {run: read_pred_csv(ROOT / f"outputs/{run}/predictions.csv") for run in RUNS}
    for run in RUNS:
        if existing[run][0] != bench_ids:
            raise ValueError(f"{run} predictions.csv IDs differ from benchmark.csv")
    reference_pred = existing[REFERENCE_RUN][1]
    results, acceptance, best = {}, {}, {}
    for run in RUNS:
        logits = load_logits(output / run)
        if logits["val"]["id"] != val_ids or logits["benchmark"]["id"] != bench_ids:
            raise ValueError(f"Saved logits IDs differ from split/benchmark: {run}")
        if any(logits[s]["labels"] != list(LABELS) for s in logits):
            raise ValueError(f"Saved label order differs: {run}")
        rows = evaluate_run(logits["val"]["logits"], val_truth, logits["benchmark"]["logits"], prior,
                            existing[run][1], reference_pred)
        results[run] = rows
        zero = next(r for r in rows if r["tau"] == 0.0)
        recorded = json.loads((ROOT / f"outputs/{run}/summary.json").read_text(encoding="utf-8"))["best_validation_macro_f1"]
        acceptance[run] = {"recorded_val_macro_f1": recorded, "tau0_val_macro_f1": zero["val_macro_f1"],
                           "val_macro_f1_match": math.isclose(recorded, zero["val_macro_f1"], rel_tol=0, abs_tol=1e-9),
                           "benchmark_diff_vs_own_csv_at_tau0": zero["benchmark_diff_vs_own_csv"]}
        acceptance[run]["passed"] = (acceptance[run]["val_macro_f1_match"]
                                     and zero["benchmark_diff_vs_own_csv"] == 0)
        taus = [r["tau"] for r in rows]
        scores = [r["val_macro_f1"] for r in rows]
        best[run] = {"tau": select_tau(taus, scores), "val_macro_f1": max(scores),
                     "flat_taus_within_0.2pp": flat_taus(taus, scores)}
        best[run]["val_macro_f1_at_tau0"] = zero["val_macro_f1"]
        best[run]["gain_vs_tau0"] = max(scores) - zero["val_macro_f1"]
        best[run]["_bench_logits"] = logits["benchmark"]["logits"]
    if not all(a["passed"] for a in acceptance.values()):
        return None, None, None, acceptance

    # Submission candidates, deduplicated against existing CSVs and each other.
    submissions, records, seen = {}, [], {f"outputs/{r}/predictions.csv": existing[r][1] for r in RUNS}
    for run in RUNS:
        tau = best[run]["tau"]
        pred = predict_adjusted(best[run]["_bench_logits"], prior, tau)
        name = f"{run}_tau{tau:+.1f}.csv"
        dup = next((p for p, q in seen.items() if np.array_equal(q, pred)), None)
        rec = {"run": run, "tau": tau, "file": None, "duplicate_of": dup,
               "diff_vs_own_csv": int(np.count_nonzero(pred != existing[run][1])),
               "diff_vs_weighted_quarter_csv": int(np.count_nonzero(pred != reference_pred)),
               "counts": {k: int(np.count_nonzero(pred == i)) for i, k in enumerate(LABELS)}}
        if dup is None:
            rec["file"] = f"{OUTPUT_REL}/submissions/{name}"
            submissions[name] = (bench_ids, pred)
            seen[rec["file"]] = pred
        records.append(rec)
    if len(submissions) > 3:
        raise ValueError("At most 3 submission CSVs allowed")
    sub_dir = output / "submissions"
    for name, (ids, pred) in submissions.items():
        if write:
            sub_dir.mkdir(exist_ok=True)
            write_pred_csv(sub_dir / name, ids, pred)
    for rec in records:
        if rec["file"]:
            path = ROOT / rec["file"]
            if write:
                check_pred_csv(path, bench_ids)
                rec["sha256"] = sha256_file(path)
            elif path.exists():
                check_pred_csv(path, bench_ids)
                rec["sha256"] = sha256_file(path)
    for run in RUNS:
        best[run].pop("_bench_logits")
    summary = {
        "schema_version": 1, "labels": list(LABELS), "formula": "adjusted = logits - tau * log(pi)",
        "training_split_rows": int(counts.sum()),
        "train_class_counts": {k: int(c) for k, c in zip(LABELS, counts)},
        "pi": {k: float(p) for k, p in zip(LABELS, prior)},
        "tau_grid": list(TAU_GRID), "selection_rule": "max validation Macro-F1; ties -> smallest |tau|, then smaller tau",
        "flat_tolerance_macro_f1": FLAT_TOLERANCE, "reference_csv": f"outputs/{REFERENCE_RUN}/predictions.csv",
        "acceptance_tau0": acceptance, "best": best, "submissions": records,
        "results": results,
    }
    return summary, table_rows(results), submissions, acceptance


def common_inputs():
    train, validation, _ = load_split(ROOT / "dataset", ROOT / "outputs/eda/split.json")
    benchmark = read_rows(ROOT / "dataset/benchmark.csv", labeled=False)
    if len(train) != 8100 or len(validation) != 900 or len(benchmark) != 1000:
        raise ValueError("Expected 8,100 / 900 / 1,000 rows")
    counts, prior = class_prior([r["label"] for r in train])
    truth = np.array([LABELS.index(r["label"]) for r in validation])
    return validation, benchmark, counts, prior, truth


def infer_logits(model, batches, torch):
    device = torch.device("cuda")
    out = []
    with torch.inference_mode():
        for batch in batches:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits  # FP32, no autocast
            if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                raise ValueError("Expected finite FP32 logits")
            out.append(logits.cpu().numpy())
    return np.concatenate(out)


def run_inference(output, validation, benchmark):
    import gc
    import platform
    import torch
    import transformers
    from src.training import load_checkpoint, loader
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required")
    meta = {"batch_size": BATCH_SIZE, "precision": "fp32 (no autocast)", "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(), "runs": {}}
    for run in RUNS:
        checkpoint = ROOT / f"outputs/{run}/checkpoint"
        model, tokenizer, config = load_checkpoint(checkpoint, torch.device("cuda"))
        if {p.dtype for p in model.parameters()} != {torch.float32}:
            raise ValueError("Expected FP32 model weights")
        if config["validation_ids"] != [r["id"] for r in validation]:
            raise ValueError("Checkpoint validation IDs differ from the fixed split")
        meta["runs"][run] = {"max_length": config["max_length"], "preprocessing": config["preprocessing"],
                             "model_sha256": sha256_file(checkpoint / "model.safetensors")}
        destination = output / run
        destination.mkdir(parents=True)
        for split, rows, labeled in (("val", validation, True), ("benchmark", benchmark, False)):
            batches = loader(rows, tokenizer, config["max_length"], BATCH_SIZE, labeled=labeled,
                             preprocessing=config["preprocessing"])
            logits = infer_logits(model, batches, torch)
            if logits.shape != (len(rows), len(LABELS)):
                raise ValueError("Logit shape mismatch")
            extra = {"truth": np.array([LABELS.index(r["label"]) for r in rows])} if labeled else {}
            np.savez(destination / f"logits_{split}.npz", id=np.array([r["id"] for r in rows]),
                     logits=logits, labels=np.array(LABELS), **extra)
            print(f"{run} {split}: {len(rows)} rows", flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    write_json(output / "meta.json", meta)


def json_roundtrip(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="recompute from saved logits and verify outputs")
    args = parser.parse_args()
    output = ROOT / OUTPUT_REL
    validation, benchmark, counts, prior, truth = common_inputs()
    val_ids, bench_ids = [r["id"] for r in validation], [r["id"] for r in benchmark]
    if args.check:
        summary, table, subs, acceptance = build(output, val_ids, bench_ids, truth, prior, counts, write=False)
        if summary is None:
            raise SystemExit(f"Acceptance failed: {json.dumps(acceptance)}")
        saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        problems = []
        if json_roundtrip(summary) != saved:
            problems.append("summary.json differs from recomputation")
        with (output / "tau_table.csv").open(encoding="utf-8", newline="") as stream:
            if list(csv.DictReader(stream)) != json_roundtrip_table(table):
                problems.append("tau_table.csv differs from recomputation")
        sub_dir = output / "submissions"
        on_disk = sorted(p.name for p in sub_dir.glob("*.csv")) if sub_dir.exists() else []
        if on_disk != sorted(subs):
            problems.append(f"submission files {on_disk} != expected {sorted(subs)}")
        for name, (ids, pred) in subs.items():
            read_ids, read_pred = read_pred_csv(sub_dir / name)
            if read_ids != ids or not np.array_equal(read_pred, pred):
                problems.append(f"submission content differs: {name}")
        if problems:
            raise SystemExit("CHECK FAILED: " + "; ".join(problems))
        print("CHECK OK: summary, table and submissions reproduce from saved logits")
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)
    run_inference(output, validation, benchmark)
    summary, table, subs, acceptance = build(output, val_ids, bench_ids, truth, prior, counts, write=True)
    if summary is None:
        write_json(output / "acceptance_failed.json", acceptance)
        raise SystemExit(f"tau=0 acceptance FAILED, no CSV produced: {json.dumps(acceptance)}")
    write_table(output / "tau_table.csv", table)
    write_json(output / "summary.json", summary)
    print(json.dumps({"acceptance": acceptance, "best": summary["best"], "submissions": summary["submissions"]},
                     indent=2), flush=True)


def json_roundtrip_table(rows):
    """CSV values are strings; stringify the recomputed table the way csv.writer does."""
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return list(csv.DictReader(io.StringIO(stream.getvalue(), newline="")))


if __name__ == "__main__":
    main()
