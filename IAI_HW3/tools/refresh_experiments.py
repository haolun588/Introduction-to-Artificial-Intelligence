"""Audit local HW3 artifacts and refresh the tracked experiment index (stdlib only)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["negative", "neutral", "positive"]
REGISTRY = "experiments/registry.json"
SNAPSHOT = "experiments/snapshot.json"
REPORT = "docs/Experiment_Results.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), f"Path outside HW3: {relative}")
    return path


def read_json(relative):
    return json.loads(local_path(relative).read_text(encoding="utf-8"))


def digest(relative):
    with local_path(relative).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(relative, fields):
    with local_path(relative).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, f"Unexpected CSV header: {relative}")
        rows = list(reader)
    require(all(set(row) == set(fields) and all(v is not None for v in row.values())
                for row in rows), f"Malformed CSV rows: {relative}")
    ids = [row["id"] for row in rows]
    require(all(ids) and len(set(ids)) == len(ids), f"Missing/duplicate IDs: {relative}")
    return rows


def check_metrics(result, expected_count):
    cm = result["confusion_matrix"]
    require(len(cm) == 3 and all(len(row) == 3 for row in cm), "Invalid confusion matrix")
    require(all(isinstance(n, int) and n >= 0 for row in cm for n in row), "Invalid counts")
    require(sum(map(sum, cm)) == expected_count, "Validation count mismatch")
    f1 = []
    for i in range(3):
        denominator = sum(cm[i]) + sum(row[i] for row in cm)
        f1.append(2 * cm[i][i] / denominator if denominator else 0.0)
    require(math.isclose(sum(f1) / 3, result["macro_f1"], abs_tol=1e-12),
            "Macro-F1 does not match confusion matrix")


def collect_cross_validations(registry, dataset_hash, benchmark_ids):
    """Audit registered k-fold CV groups (fold models, OOF metrics, ensemble CSV); files only, no inference."""
    entries = registry.get("cross_validations", [])
    ids = [entry["id"] for entry in entries]
    require(len(ids) == len(set(ids)), "Duplicate cross-validation IDs")
    if not entries:
        return []
    dataset_rows = read_csv("dataset/dataset.csv", ["id", "text", "label"])
    all_ids = [row["id"] for row in dataset_rows]
    truth = {row["id"]: row["label"] for row in dataset_rows}
    model_files = ["config.json", "training_config.json", "model.safetensors",
                   "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
    results = []
    for entry in entries:
        name = entry["id"]
        require(local_path(entry["report"]).is_file(), f"Missing CV report: {name}")
        score = entry["kaggle_macro_f1"]
        require(score is None or 0 <= score <= 1, f"Invalid Kaggle score: {name}")
        manifest_path = f"{entry['fold_dir']}/folds_manifest.json"
        manifest = read_json(manifest_path)
        require(manifest["dataset_sha256"] == dataset_hash, f"CV folds built from another dataset: {name}")
        require(manifest["k"] == len(manifest["folds"]) == len(entry["folds"]) >= 2, f"Fold count mismatch: {name}")
        seen_validation, folds = set(), []
        for fold_entry, listed in zip(entry["folds"], manifest["folds"]):
            index = fold_entry["fold"]
            require(index == listed["fold"], f"Fold order differs: {name}")
            require(fold_entry["split"] == f"{entry['fold_dir']}/{listed['file']}", f"Fold file path differs: {name}/{index}")
            split_hash = digest(fold_entry["split"])
            require(split_hash == listed["sha256"], f"Fold file differs from manifest: {name}/{index}")
            fold = read_json(fold_entry["split"])
            train_ids, val_ids = fold["train_ids"], fold["validation_ids"]
            require(fold["dataset_sha256"] == dataset_hash and fold["fold"] == index, f"Fold provenance differs: {name}/{index}")
            require(not (set(train_ids) & set(val_ids)) and set(train_ids) | set(val_ids) == set(all_ids)
                    and len(train_ids) + len(val_ids) == len(all_ids), f"Invalid fold partition: {name}/{index}")
            require(not (seen_validation & set(val_ids)), f"Validation folds overlap: {name}/{index}")
            seen_validation |= set(val_ids)
            checkpoint, output = fold_entry["checkpoint"], fold_entry["output"]
            config = read_json(f"{checkpoint}/training_config.json")
            require(config["labels"] == LABELS and not config["smoke"],
                    f"Invalid labels or smoke checkpoint: {checkpoint}")
            require(config["dataset_sha256"] == dataset_hash and config["split_sha256"] == split_hash,
                    f"Dataset/split mismatch: {checkpoint}")
            excluded = (config.get("exclude_train_ids") or {}).get("removed_train_ids", [])
            require(set(excluded) <= set(train_ids) and len(set(excluded)) == len(excluded)
                    and config["train_ids"] == [i for i in train_ids if i not in set(excluded)]
                    and config["validation_ids"] == val_ids,
                    f"Split IDs differ from fold file: {checkpoint}")
            init_from = config.get("init_from")
            if init_from:  # runs fine-tuned from an MLM checkpoint: the checkpoint must still exist unchanged
                mlm_dir = Path(init_from["path"])
                mlm_rel = mlm_dir.relative_to(ROOT).as_posix() if mlm_dir.is_absolute() else mlm_dir.as_posix()
                require(local_path(f"{mlm_rel}/mlm_summary.json").is_file()
                        and local_path(f"{mlm_rel}/model.safetensors").is_file(),
                        f"Missing MLM init checkpoint files: {mlm_rel}")
                require(digest(f"{mlm_rel}/model.safetensors") == init_from["model_sha256"],
                        f"MLM init checkpoint hash differs: {mlm_rel}")
            check_metrics(config["validation"], len(val_ids))
            model = read_json(f"{checkpoint}/config.json")
            require(model["id2label"] == {str(i): label for i, label in enumerate(LABELS)},
                    f"Model labels differ: {checkpoint}")
            summary = read_json(f"{output}/summary.json")
            run_config = read_json(f"{output}/run_config.json")
            require(not run_config["smoke"] and not summary["smoke"], f"Smoke run in CV: {output}")
            require(all(config.get(k) == v for k, v in run_config.items()),
                    f"Run/checkpoint configs differ: {output}")
            result = summary["reloaded_validation"]
            require(summary["reload_verified"] and result["confusion_matrix"] == config["validation"]["confusion_matrix"],
                    f"Saved reload verification differs: {output}")
            require(summary["best_epoch"] == config["best_epoch"] and
                    math.isclose(summary["best_validation_macro_f1"], result["macro_f1"], abs_tol=1e-12),
                    f"Best checkpoint metadata differs: {output}")
            folds.append({**fold_entry, "split_sha256": split_hash, "best_epoch": config["best_epoch"],
                          "validation_macro_f1": result["macro_f1"], "training_seconds": summary.get("total_seconds"),
                          "checkpoint_files_sha256": {f: digest(f"{checkpoint}/{f}") for f in model_files},
                          "summary_sha256": digest(f"{output}/summary.json")})
        require(seen_validation == set(all_ids), f"Validation folds do not cover the dataset: {name}")
        metrics_path = f"{entry['oof_output']}/oof_metrics.json"
        oof = read_json(metrics_path)
        require(oof["rows"] == len(all_ids) and oof["dataset_sha256"] == dataset_hash and oof["labels"] == LABELS
                and oof["k"] == manifest["k"], f"OOF metrics provenance differs: {name}")
        check_metrics(oof["oof"], oof["rows"])
        require(len(oof["folds"]["per_fold"]) == manifest["k"], f"OOF fold count differs: {name}")
        predictions_path = f"{entry['oof_output']}/oof_predictions.csv"
        rows = read_csv(predictions_path, ["id", "label", "pred", "logit_negative", "logit_neutral", "logit_positive"])
        require([r["id"] for r in rows] == all_ids and all(r["label"] == truth[r["id"]] for r in rows)
                and all(r["pred"] in LABELS for r in rows), f"OOF prediction IDs/labels differ: {name}")
        matrix = [[sum(r["label"] == a and r["pred"] == b for r in rows) for b in LABELS] for a in LABELS]
        require(matrix == oof["oof"]["confusion_matrix"], f"OOF CSV differs from metrics: {name}")
        submission = entry["submission"]
        sub_rows = read_csv(submission["path"], ["id", "pred"])
        require(len(sub_rows) == 1000 and [r["id"] for r in sub_rows] == benchmark_ids
                and all(r["pred"] in LABELS for r in sub_rows), f"CV submission IDs/labels differ: {name}")
        require(digest(submission["path"]) == submission["sha256"] == oof["ensemble"]["sha256"],
                f"CV submission hash differs: {name}")
        require(1 <= len(oof["ensemble"]["folds"]) <= 3, f"CV ensemble exceeds 3 models: {name}")
        results.append({**entry, "folds": folds, "oof": oof["oof"], "fold_scores": oof["folds"],
                        "original_validation_in_oof": oof.get("original_900_validation_ids_in_oof"),
                        "ensemble_folds": oof["ensemble"]["folds"], "submission_rows": len(sub_rows),
                        "source_sha256": {metrics_path: digest(metrics_path), predictions_path: digest(predictions_path),
                                          manifest_path: digest(manifest_path)}})
    by_id = {result["id"]: result for result in results}
    for result in results:
        comparison = result.get("comparison")
        if comparison is None:
            continue
        name = result["id"]
        require(comparison["b"] == name and comparison["a"] in by_id and comparison["a"] != name,
                f"Invalid CV comparison references: {name}")
        a, b = by_id[comparison["a"]], result
        report = read_json(comparison["path"])
        require(report["dataset_sha256"] == dataset_hash and report["k"] == len(b["folds"]) == len(a["folds"])
                and report["rows"] == len(all_ids), f"CV comparison provenance differs: {name}")
        for side, cv in (("A", a), ("B", b)):
            require(math.isclose(report[side]["oof_macro_f1"], cv["oof"]["macro_f1"], abs_tol=1e-9),
                    f"CV comparison {side} OOF Macro-F1 differs from registry CV: {name}")
        require(math.isclose(report["delta_oof_macro_f1"], b["oof"]["macro_f1"] - a["oof"]["macro_f1"], abs_tol=1e-9),
                f"CV comparison delta inconsistent: {name}")
        boot = report["paired_bootstrap"]
        require(boot["delta_p2_5"] <= boot["delta_p97_5"], f"CV comparison interval invalid: {name}")
        comparison.update(sha256=digest(comparison["path"]), delta=report["delta_oof_macro_f1"],
                          delta_class_f1=report["delta_class_f1"], ci_low=boot["delta_p2_5"],
                          ci_high=boot["delta_p97_5"], fraction_delta_le_0=boot["fraction_delta_le_0"],
                          resamples=boot["resamples"], bootstrap_seed=boot["seed"],
                          b_wins_folds=report["per_fold"]["b_wins"], disagreement=report["disagreement"])
    return results


def collect():
    registry = read_json(REGISTRY)
    require(registry["schema_version"] == 1, "Unsupported registry version")
    entries = registry["runs"]
    ids = [entry["id"] for entry in entries]
    require(len(ids) == len(set(ids)), "Duplicate experiment IDs")
    roles = ("validation_reference", "comparison_control", "kaggle_candidate", "release_candidate")
    for role in roles:
        require(registry.get(role) is None or registry[role] in ids, f"Unknown {role}")
    benchmark = read_csv("dataset/benchmark.csv", ["id", "text"])
    benchmark_ids = [row["id"] for row in benchmark]
    require(len(benchmark) == 1000, "Expected 1,000 benchmark rows")
    split_path = "outputs/eda/split.json"
    split = read_json(split_path)
    dataset_hash, split_hash = digest("dataset/dataset.csv"), digest(split_path)
    require(split["dataset_sha256"] == dataset_hash, "Dataset differs from fixed split")
    validation_count = len(split["validation_ids"])
    confirmation_splits = []
    if registry.get("confirmation_splits"):
        dataset_rows = read_csv("dataset/dataset.csv", ["id", "text", "label"])
        by_id = {r["id"]: r for r in dataset_rows}
        for entry in registry["confirmation_splits"]:
            extra = read_json(entry["path"])
            require(digest(entry["path"]) == entry["sha256"] and extra["seed"] == entry["seed"]
                    and extra["dataset_sha256"] == dataset_hash, "Confirmation split provenance differs")
            train_ids, val_ids = extra["train_ids"], extra["validation_ids"]
            require(len(train_ids) == 8100 and len(val_ids) == 900
                    and len(set(train_ids + val_ids)) == len(by_id)
                    and set(train_ids + val_ids) == set(by_id), "Invalid confirmation split partition")
            train_text = {" ".join(by_id[i]["text"].split()) for i in train_ids}
            val_text = {" ".join(by_id[i]["text"].split()) for i in val_ids}
            require(not (train_text & val_text), "Text leakage in confirmation split")
            for name, part in (("train", train_ids), ("validation", val_ids)):
                counts = {label: sum(by_id[i]["label"] == label for i in part) for label in LABELS}
                require(all(counts.values()) and counts == extra["class_counts"][name], "Confirmation class counts differ")
            confirmation_splits.append({**entry, "train_rows": len(train_ids), "validation_rows": len(val_ids),
                                        "class_counts": extra["class_counts"],
                                        "overlap_with_primary_validation": len(set(val_ids) & set(split["validation_ids"]))})
    checkpoint_info, configs, predictions, runs = {}, {}, {}, []
    for entry in entries:
        require(local_path(entry["report"]).is_file(), f"Missing report: {entry['id']}")
        score = entry["kaggle_macro_f1"]
        require(score is None or 0 <= score <= 1, "Invalid Kaggle score")
        for checkpoint in entry["checkpoints"]:
            if checkpoint in configs:
                continue
            config = read_json(f"{checkpoint}/training_config.json")
            require(config["labels"] == LABELS and not config["smoke"],
                    f"Invalid labels or smoke checkpoint: {checkpoint}")
            require(config["dataset_sha256"] == dataset_hash and config["split_sha256"] == split_hash,
                    f"Dataset/split mismatch: {checkpoint}")
            require(config["train_ids"] == split["train_ids"] and
                    config["validation_ids"] == split["validation_ids"], f"Split IDs differ: {checkpoint}")
            check_metrics(config["validation"], validation_count)
            model = read_json(f"{checkpoint}/config.json")
            require(model["id2label"] == {str(i): name for i, name in enumerate(LABELS)},
                    f"Model labels differ: {checkpoint}")
            required = ["config.json", "training_config.json", "model.safetensors",
                        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
            checkpoint_info[checkpoint] = {
                "files_sha256": {name: digest(f"{checkpoint}/{name}") for name in required},
                "config": {k: v for k, v in config.items() if k not in ("train_ids", "validation_ids")},
            }
            configs[checkpoint] = config
        output = entry["output"]
        summary = read_json(f"{output}/summary.json")
        source_hashes = {f"{output}/summary.json": digest(f"{output}/summary.json")}
        if entry["kind"] == "single":
            require(len(entry["checkpoints"]) == 1, "Single run must have one checkpoint")
            config = configs[entry["checkpoints"][0]]
            run_config = read_json(f"{output}/run_config.json")
            require(all(config.get(k) == v for k, v in run_config.items()),
                    f"Run/checkpoint configs differ: {entry['id']}")
            result = summary["reloaded_validation"]
            require(summary["reload_verified"] and result["confusion_matrix"] == config["validation"]["confusion_matrix"],
                    f"Saved reload verification differs: {entry['id']}")
            require(summary["best_epoch"] == config["best_epoch"] and
                    math.isclose(summary["best_validation_macro_f1"], result["macro_f1"], abs_tol=1e-12),
                    f"Best checkpoint metadata differs: {entry['id']}")
            source_hashes[f"{output}/run_config.json"] = digest(f"{output}/run_config.json")
            if "eval_interval" in run_config:
                history = read_json(f"{output}/history.json")
                expected_epochs = [epoch + fraction for epoch in range(run_config["epochs"])
                                   for fraction in run_config["evaluation_nominal_fractions"]]
                require([row["epoch"] for row in history] == expected_epochs, "Evaluation schedule incomplete")
                require(summary["evaluation_count"] == len(history), "Evaluation count differs")
                for row in history:
                    check_metrics(row["train"], len(split["train_ids"]))
                    check_metrics(row["validation"], validation_count)
                    require(row["scheduler_steps"] == row["optimizer_updates"]
                            and row["skipped_updates"] == row["attempted_updates"] - row["optimizer_updates"],
                            "Optimizer/scheduler counts differ")
                best = max(history, key=lambda row: row["validation"]["macro_f1"])
                require(best["epoch"] == config["best_epoch"] and best["validation"] == config["validation"],
                        "Selected checkpoint differs from earliest best evaluation")
                require(history[-1]["attempted_updates"] == run_config["optimizer_steps_planned"]
                        and summary["optimizer_updates"] == history[-1]["optimizer_updates"],
                        "Training update count incomplete")
                source_hashes.update({f"{output}/{name}": digest(f"{output}/{name}")
                                      for name in ("history.json", "curves.png")})
            for relative, expected in summary.get("recovery_evidence_sha256", {}).items():
                require(digest(relative) == expected, "Recovered training evidence changed")
                source_hashes[relative] = expected
            comparison_path = f"{output}/control_comparison.json"
            if local_path(comparison_path).exists():
                source_hashes[comparison_path] = digest(comparison_path)
        else:
            require(entry["kind"] == "ensemble", "Unknown experiment kind")
            # The original learning-rate comparison predates the optional names field.
            names = summary.get("names", ["lr2e5", "lr3e5"])
            require(len(names) == len(entry["checkpoints"]), "Ensemble size mismatch")
            require(len(summary["checkpoints"]) == len(entry["checkpoints"]) and all(
                saved.replace("\\", "/").endswith("/" + relative)
                for saved, relative in zip(summary["checkpoints"], entry["checkpoints"])),
                f"Ensemble checkpoint paths differ: {entry['id']}")
            require(summary["dataset_sha256"] == dataset_hash and summary["split_sha256"] == split_hash
                    and summary["validation_ids"] == split["validation_ids"] and summary["labels"] == LABELS,
                    f"Ensemble provenance mismatch: {entry['id']}")
            for name, checkpoint in zip(names, entry["checkpoints"]):
                require(summary["metrics"][name]["confusion_matrix"] == configs[checkpoint]["validation"]["confusion_matrix"],
                        f"Ensemble member differs: {entry['id']} / {name}")
            result = summary["metrics"]["soft_voting"]
        check_metrics(result, validation_count)
        prediction_path = f"{output}/predictions.csv"
        rows = read_csv(prediction_path, ["id", "pred"])
        require([row["id"] for row in rows] == benchmark_ids, f"Benchmark ID order differs: {entry['id']}")
        require(all(row["pred"] in LABELS for row in rows), f"Invalid predicted label: {entry['id']}")
        predictions[entry["id"]] = [row["pred"] for row in rows]
        historical_score = local_path(f"{output}/kaggle_result.json")
        if historical_score.exists():
            require(read_json(f"{output}/kaggle_result.json")["macro_f1"] == score,
                    f"Kaggle record differs: {entry['id']}")
        runs.append({**entry, "validation": result, "best_epoch": summary.get("best_epoch"),
                     "training_seconds": summary.get("total_seconds"), "source_sha256": source_hashes,
                     "predictions": prediction_path, "predictions_sha256": digest(prediction_path),
                     "prediction_rows": len(rows)})
    candidate = predictions[registry["kaggle_candidate"]]
    for run in runs:
        run["different_predictions_from_kaggle_candidate"] = sum(
            a != b for a, b in zip(predictions[run["id"]], candidate))
    audits = []
    for entry in registry.get("audits", []):
        require(entry["kind"] == "precision", "Unknown audit kind")
        require(local_path(entry["report"]).is_file(), "Missing audit report")
        summary_path = f"{entry['output']}/summary.json"
        summary = read_json(summary_path)
        require(summary["completed"] and summary["historical_reproduction_passed"], "Incomplete precision audit")
        require(summary["labels"] == LABELS and summary["validation_ids"] == split["validation_ids"]
                and summary["benchmark_ids"] == benchmark_ids, "Audit IDs/labels differ")
        require(summary["source_sha256"]["dataset/dataset.csv"] == dataset_hash
                and summary["source_sha256"][split_path] == split_hash
                and summary["source_sha256"]["dataset/benchmark.csv"] == digest("dataset/benchmark.csv"),
                "Audit data/split differs")
        require(set(summary["runs"]) == set(entry["run_ids"]), "Audit run IDs differ")
        for run_id, result in summary["runs"].items():
            require(run_id in predictions, "Unknown audited candidate")
            require(result["checkpoint_files_sha256"] == checkpoint_info[result["checkpoint"]]["files_sha256"],
                    "Audited checkpoint changed")
            require(result["saved_predictions_sha256"] == digest(result["saved_predictions"]), "Audited CSV changed")
            for mode in ("amp", "fp32"):
                check_metrics(result["validation"][mode]["metrics"], validation_count)
                rows = read_csv(f"{entry['output']}/{run_id}/benchmark/predictions_{mode}.csv", ["id", "pred"])
                require([r["id"] for r in rows] == benchmark_ids and all(r["pred"] in LABELS for r in rows),
                        "Audit prediction IDs/labels differ")
                differences = sum(r["pred"] != old for r, old in zip(rows, predictions[run_id]))
                require(differences == result["benchmark"]["differences_from_saved_csv"][mode],
                        "Audit saved CSV comparison differs")
            require(result["validation"]["amp_reproduces_saved_validation"]
                    and result["validation"]["amp"]["metrics"]["confusion_matrix"]
                    == configs[result["checkpoint"]]["validation"]["confusion_matrix"]
                    and result["benchmark"]["differences_from_saved_csv"]["fp32"] == 0,
                    "Audit does not reproduce historical results")
        for relative, expected in summary["artifacts_sha256"].items():
            require(digest(f"{entry['output']}/{relative}") == expected, "Audit artifact hash differs")
        audits.append({**entry, "summary_sha256": digest(summary_path),
                       "results": {k: v for k, v in summary.items()
                                   if k not in ("validation_ids", "benchmark_ids")}})
    packages = []
    for entry in registry.get("package_rehearsals", []):
        require(entry["run_id"] in predictions and local_path(entry["report"]).is_file(), "Invalid package entry")
        result = read_json(entry["summary"])
        require(result["candidate"] == entry["run_id"] and result["clean_environment_verified"]
                and result["prediction_verified"], "Package rehearsal incomplete")
        require(digest(result["archive"]) == result["archive_sha256"], "Package ZIP changed")
        package = result["extracted"]
        require(digest(f"{package}/package_manifest.json") == result["manifest_sha256"], "Package manifest changed")
        manifest = read_json(f"{package}/package_manifest.json")
        require(manifest["candidate"] == entry["run_id"], "Packaged candidate differs")
        for relative, info in manifest["files"].items():
            require(digest(f"{package}/{relative}") == info["sha256"], f"Packaged file changed: {relative}")
        for name, checksum in checkpoint_info[manifest["source_checkpoint"]]["files_sha256"].items():
            require(manifest["files"][f"saved_models/checkpoint/{name}"]["sha256"] == checksum,
                    "Packaged checkpoint differs from registered model")
        require(manifest["files"]["dataset/dataset.csv"]["sha256"] == dataset_hash
                and manifest["files"]["dataset/benchmark.csv"]["sha256"] == digest("dataset/benchmark.csv")
                and manifest["files"][split_path]["sha256"] == split_hash, "Packaged data/split differs")
        rows = read_csv(result["predictions"], ["id", "pred"])
        require([r["id"] for r in rows] == benchmark_ids
                and [r["pred"] for r in rows] == predictions[entry["run_id"]], "Packaged predictions differ")
        require(digest(result["predictions"]) == manifest["reference_predictions_sha256"]
                == result["predictions_sha256"], "Packaged prediction hash differs")
        for relative, checksum in result["evidence_sha256"].items():
            require(digest(relative) == checksum, "Package verification evidence changed")
        packages.append({**entry, "summary_sha256": digest(entry["summary"]), "results": result})
    calibrations = []
    for entry in registry.get("calibrations", []):
        result = read_json(entry["summary"])
        require(local_path(entry["report"]).is_file(), "Missing calibration report")
        require(result["completed"] and result["clean_inference_unchanged"]
                and result["optimizer_updates"] == 0, "Incomplete calibration audit")
        require(result["dataset_sha256"] == dataset_hash and result["split_sha256"] == split_hash,
                "Calibration data/split differs")
        require(set(result["scale_train_ids"] + result["audit_train_ids"]).issubset(split["train_ids"]),
                "Calibration must use only training rows")
        require(len(entry["smoke_outputs"]) == len(result["epsilons"]) <= 2, "Calibration budget mismatch")
        evidence = {entry["summary"]: digest(entry["summary"])}
        for output, epsilon in zip(entry["smoke_outputs"], result["epsilons"]):
            config, summary = read_json(f"{output}/run_config.json"), read_json(f"{output}/summary.json")
            require(config["smoke"] and summary["smoke"] and summary["reload_verified"], "Incomplete smoke run")
            require(config["fgsm"]["epsilon"] == epsilon and config["split_sha256"] == split_hash,
                    "Smoke epsilon/split differs from calibration")
            require(summary["optimizer_updates"] == summary["scheduler_steps"] > 0
                    and summary["skipped_updates"] == 0, "Unexpected smoke optimizer updates")
            files = [local_path(f"{output}/{name}") for name in
                     ("run_config.json", "summary.json", "history.json", "curves.png", "environment.json", "split.json")]
            require(all(path.is_file() for path in files), "Missing smoke evidence")
            require(all(local_path(f"{output}/checkpoint/{name}").is_file() for name in
                        ("model.safetensors", "config.json", "training_config.json", "tokenizer.json")),
                    "Missing smoke checkpoint")
            files += sorted(local_path(f"{output}/checkpoint").glob("*"))
            for path in files:
                if path.is_file():
                    relative = path.relative_to(ROOT).as_posix()
                    evidence[relative] = digest(relative)
        calibrations.append({**entry, "results": result, "evidence_sha256": evidence})
    cross_validations = collect_cross_validations(registry, dataset_hash, benchmark_ids)
    code = ["main.py", "src/__init__.py", "src/training.py", "src/eda.py", "src/preprocessing.py",
            "src/model_source.py", "src/mlm.py", "tools/__init__.py", "tools/analyze_ensemble.py", "tools/audit_emoji.py",
            "tools/audit_precision.py", "tools/cv_oof.py", "tools/refresh_experiments.py", "tools/calibrate_fgsm.py",
            "tools/package_candidate.py", "tools/verify_package.py",
            "requirements.txt", "requirements/train.txt", "requirements/eda.txt", "requirements/lock.txt"]
    return {"schema_version": 1, "reviewed_on": registry["reviewed_on"],
            "roles": {k: registry.get(k) for k in roles},
            "score_provenance": registry["score_provenance"], "registry_sha256": digest(REGISTRY),
            "dataset_sha256": dataset_hash, "benchmark_sha256": digest("dataset/benchmark.csv"),
            "split_sha256": split_hash, "train_rows": len(split["train_ids"]), "validation_rows": validation_count,
            "audit_scope": "File/metadata/CSV checks only; no model inference or platform verification. Current code hashes do not establish original training code provenance.",
            "current_code_sha256": {name: digest(name) for name in code},
            "checkpoints": checkpoint_info, "runs": runs, "audits": audits, "package_rehearsals": packages,
            "confirmation_splits": confirmation_splits, "calibrations": calibrations,
            "cross_validations": cross_validations}


def render(snapshot):
    lines = ["# HW3 實驗總表", "", "由 `python tools/refresh_experiments.py` 產生；不要手動修改此檔。",
             "更新來源為 [registry.json](../experiments/registry.json)，完整雜湊與指標見 [snapshot.json](../experiments/snapshot.json)。",
             "", "驗證分數讀取本機 summary／checkpoint；Kaggle 分數來自歷史使用者回報，未登入平台核驗。`待回報` 不等於未提交。",
             "", "| 實驗 | 驗證 Macro-F1 | Kaggle | 與 Kaggle 候選不同筆數 | 決策 |",
             "|---|---:|---:|---:|---|"]
    for run in snapshot["runs"]:
        score = ("未新增提交" if run.get("submission_status") == "duplicate_not_submitted" else "待回報") if run["kaggle_macro_f1"] is None else f"{run['kaggle_macro_f1']:.4f}"
        lines.append(f"| [{run['name']}](../{run['report']}) | {run['validation']['macro_f1']:.2%} | {score} | "
                     f"{run['different_predictions_from_kaggle_candidate']} | {run['decision']} |")
    lines += ["", "## 模型與提交檔", "", "所有路徑相對於 HW3 根目錄；每份 CSV 均核對 1,000 筆、`id,pred`、唯一 ID、原始順序與合法標籤。",
              "模型檔、tokenizer 與設定逐檔保存 SHA-256；總表工具僅核對檔案，另行推論的證據列於下方稽核紀錄。", "",
              "| 實驗 ID | checkpoint | CSV | CSV SHA-256（前 12 碼） |", "|---|---|---|---|"]
    for run in snapshot["runs"]:
        checkpoints = "<br>".join(f"`{p}`" for p in run["checkpoints"])
        lines.append(f"| `{run['id']}` | {checkpoints} | [predictions.csv](../{run['predictions']}) | `{run['predictions_sha256'][:12]}` |")
    if snapshot["audits"]:
        lines += ["", "## 推論稽核（不新增訓練候選）", ""]
        for audit in snapshot["audits"]:
            lines.append(f"- [{audit['id']}](../{audit['report']})：已完成；歷史驗證與 FP32 提交 CSV 重現通過。")
            for name, result in audit["results"]["runs"].items():
                val, bench = result["validation"], result["benchmark"]
                lines.append(f"  - `{name}`：AMP／FP32 驗證 F1 {val['amp']['metrics']['macro_f1']:.4%}／"
                             f"{val['fp32']['metrics']['macro_f1']:.4%}；驗證／benchmark 標籤差異 "
                             f"{val['comparison']['label_disagreements']}／{bench['comparison']['label_disagreements']} 筆。")
    if snapshot["package_rehearsals"]:
        lines += ["", "## 候選封裝演練", ""]
        for package in snapshot["package_rehearsals"]:
            result = package["results"]
            lines.append(f"- [{package['run_id']} 獨立封裝](../{package['report']})：新 venv 安裝、乾淨解壓推論通過，"
                         "1,000 筆與原 CSV 完全相同；此為演練，尚非最終交件。")
            lines.append(f"- [演練 ZIP](../{result['archive']})；SHA-256 前 12 碼 `{result['archive_sha256'][:12]}`。")
    if snapshot["calibrations"]:
        lines += ["", "## 方法校準（不列入正式候選）", ""]
        for entry in snapshot["calibrations"]:
            lines.append(f"- [{entry['id']}](../{entry['report']})：訓練資料尺度校準及兩組 smoke 完成；精度差異與限制見報告，尚無完整訓練成效。")
    for cv in snapshot["cross_validations"]:
        stats, classes = cv["fold_scores"], cv["oof"]["classes"]
        total = sum(map(sum, cv["oof"]["confusion_matrix"]))
        score = "待回報" if cv["kaggle_macro_f1"] is None else f"{cv['kaggle_macro_f1']:.4f}"
        lines += ["", "## 交叉驗證（OOF；不列入上方單模型比較）", "",
                  f"- [{cv['name']}](../{cv['report']})：{len(cv['folds'])} 折、{total:,} 筆 OOF；"
                  f"OOF Macro-F1 **{cv['oof']['macro_f1']:.2%}**，fold 平均 {stats['mean']:.2%} ± {stats['sample_std'] * 100:.2f} 個百分點（樣本標準差）。",
                  "- 三類 F1：" + "／".join(f"{label} {classes[label]['f1']:.2%}" for label in LABELS) + "。",
                  f"- 集成 CSV：`{cv['submission']['path']}`（fold {'、'.join(map(str, cv['ensemble_folds']))} 等權；"
                  f"SHA-256 前 12 碼 `{cv['submission']['sha256'][:12]}`）；提交狀態 `{cv['submission']['submission_status']}`，Kaggle {score}。"]
        if cv.get("comparison"):
            c = cv["comparison"]
            lines.append(f"- 配對比較（`{c['b']}` − `{c['a']}`，[結果](../{c['path']})）：Δ OOF Macro-F1 {c['delta'] * 100:+.2f} 個百分點，"
                         f"配對 bootstrap 95% 區間 [{c['ci_low'] * 100:+.2f}, {c['ci_high'] * 100:+.2f}]（{c['resamples']:,} 次，seed {c['bootstrap_seed']}），"
                         f"Δ≤0 比例 {c['fraction_delta_le_0']:.1%}；{c['b']} 勝 {c['b_wins_folds']}/{len(cv['folds'])} 折。")
    lines += ["", "## 比較與交件狀態", "",
              f"- 共用固定切分：{snapshot['train_rows']:,} train／{snapshot['validation_rows']:,} validation。",
              f"- 驗證參考：`{snapshot['roles']['validation_reference']}`；Kaggle 候選：`{snapshot['roles']['kaggle_candidate']}`。",
              f"- 後續新方法共同控制組：`{snapshot['roles']['comparison_control'] or '尚未完成'}`。",
              f"- 最終交件版本：`{snapshot['roles']['release_candidate'] or '尚未指定'}`。",
              "- `saved_models/checkpoint/` 仍是 `baseline`；不要直接把它當成最佳交件模型。",
              "- `ensemble_lr2_lr3/` 是同組模型的初次分析；總表採用帶 CSV 的 `ensemble_lr2_lr3_submission/`，避免重複計數。",
              "- `smoke/` 僅驗證管線，`emoji_audit/`、`emoji_short_audit/` 僅盤點，不列入正式模型比較。",
              "- 相同 Kaggle 分數不代表 CSV 相同；不同筆數以目前 Kaggle 單模型候選為基準。", "",
              "- 後續確認切分已預先固定：" + ("、".join(f"seed {s['seed']}（{s['train_rows']}／{s['validation_rows']}）" for s in snapshot['confirmation_splits']) or "尚未建立") + "；尚未在額外切分訓練。", "",
              "後續優先順序見 [92% 計畫](Target_92_Plan.md)，操作入口見 [README](../README.md)。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify artifacts and fail if the tracked index is stale; write nothing")
    args = parser.parse_args()
    try:
        snapshot = collect()
        outputs = {SNAPSHOT: json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", REPORT: render(snapshot)}
        if args.check:
            for relative, content in outputs.items():
                require(local_path(relative).is_file() and local_path(relative).read_text(encoding="utf-8") == content,
                        f"Index stale: {relative}; run python tools/refresh_experiments.py")
        else:
            for relative, content in outputs.items():
                local_path(relative).write_text(content, encoding="utf-8", newline="\n")
        print(f"Verified {len(snapshot['runs'])} runs, {len(snapshot['checkpoints'])} checkpoints, and all prediction CSVs.")
        if snapshot["cross_validations"]:
            print(f"Verified {len(snapshot['cross_validations'])} cross-validation group(s).")
        print("Index is current." if args.check else f"Updated {SNAPSHOT} and {REPORT}.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")


if __name__ == "__main__":
    main()
