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


def collect():
    registry = read_json(REGISTRY)
    require(registry["schema_version"] == 1, "Unsupported registry version")
    entries = registry["runs"]
    ids = [entry["id"] for entry in entries]
    require(len(ids) == len(set(ids)), "Duplicate experiment IDs")
    for role in ("validation_reference", "kaggle_candidate", "release_candidate"):
        require(registry[role] is None or registry[role] in ids, f"Unknown {role}")
    benchmark = read_csv("dataset/benchmark.csv", ["id", "text"])
    benchmark_ids = [row["id"] for row in benchmark]
    require(len(benchmark) == 1000, "Expected 1,000 benchmark rows")
    split_path = "outputs/eda/split.json"
    split = read_json(split_path)
    dataset_hash, split_hash = digest("dataset/dataset.csv"), digest(split_path)
    require(split["dataset_sha256"] == dataset_hash, "Dataset differs from fixed split")
    validation_count = len(split["validation_ids"])
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
    code = ["main.py", "src/__init__.py", "src/training.py", "src/eda.py", "src/preprocessing.py",
            "src/model_source.py", "tools/__init__.py", "tools/analyze_ensemble.py", "tools/audit_emoji.py",
            "requirements.txt", "requirements/train.txt", "requirements/eda.txt", "requirements/lock.txt"]
    return {"schema_version": 1, "reviewed_on": registry["reviewed_on"],
            "roles": {k: registry[k] for k in ("validation_reference", "kaggle_candidate", "release_candidate")},
            "score_provenance": registry["score_provenance"], "registry_sha256": digest(REGISTRY),
            "dataset_sha256": dataset_hash, "benchmark_sha256": digest("dataset/benchmark.csv"),
            "split_sha256": split_hash, "train_rows": len(split["train_ids"]), "validation_rows": validation_count,
            "audit_scope": "File/metadata/CSV checks only; no model inference or platform verification. Current code hashes do not establish original training code provenance.",
            "current_code_sha256": {name: digest(name) for name in code},
            "checkpoints": checkpoint_info, "runs": runs}


def render(snapshot):
    lines = ["# HW3 實驗總表", "", "由 `python tools/refresh_experiments.py` 產生；不要手動修改此檔。",
             "更新來源為 [registry.json](../experiments/registry.json)，完整雜湊與指標見 [snapshot.json](../experiments/snapshot.json)。",
             "", "驗證分數讀取本機 summary／checkpoint；Kaggle 分數來自歷史使用者回報，未登入平台核驗。`待回報` 不等於未提交。",
             "", "| 實驗 | 驗證 Macro-F1 | Kaggle | 與 Kaggle 候選不同筆數 | 決策 |",
             "|---|---:|---:|---:|---|"]
    for run in snapshot["runs"]:
        score = "待回報" if run["kaggle_macro_f1"] is None else f"{run['kaggle_macro_f1']:.4f}"
        lines.append(f"| [{run['name']}](../{run['report']}) | {run['validation']['macro_f1']:.2%} | {score} | "
                     f"{run['different_predictions_from_kaggle_candidate']} | {run['decision']} |")
    lines += ["", "## 模型與提交檔", "", "所有路徑相對於 HW3 根目錄；每份 CSV 均核對 1,000 筆、`id,pred`、唯一 ID、原始順序與合法標籤。",
              "模型檔、tokenizer 與設定逐檔保存 SHA-256；本次為檔案核對，沒有重新執行推論。", "",
              "| 實驗 ID | checkpoint | CSV | CSV SHA-256（前 12 碼） |", "|---|---|---|---|"]
    for run in snapshot["runs"]:
        checkpoints = "<br>".join(f"`{p}`" for p in run["checkpoints"])
        lines.append(f"| `{run['id']}` | {checkpoints} | [predictions.csv](../{run['predictions']}) | `{run['predictions_sha256'][:12]}` |")
    lines += ["", "## 比較與交件狀態", "",
              f"- 共用固定切分：{snapshot['train_rows']:,} train／{snapshot['validation_rows']:,} validation。",
              f"- 驗證參考：`{snapshot['roles']['validation_reference']}`；Kaggle 候選：`{snapshot['roles']['kaggle_candidate']}`。",
              f"- 最終交件版本：`{snapshot['roles']['release_candidate'] or '尚未指定'}`。",
              "- `saved_models/checkpoint/` 仍是 `baseline`；不要直接把它當成最佳交件模型。",
              "- `ensemble_lr2_lr3/` 是同組模型的初次分析；總表採用帶 CSV 的 `ensemble_lr2_lr3_submission/`，避免重複計數。",
              "- `smoke/` 僅驗證管線，`emoji_audit/`、`emoji_short_audit/` 僅盤點，不列入正式模型比較。",
              "- 相同 Kaggle 分數不代表 CSV 相同；不同筆數以目前 Kaggle 單模型候選為基準。", "",
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
        print("Index is current." if args.check else f"Updated {SNAPSHOT} and {REPORT}.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")


if __name__ == "__main__":
    main()
