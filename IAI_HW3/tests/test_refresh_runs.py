import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import refresh_experiments as refresh

LABELS = ["negative", "neutral", "positive"]
MODEL_FILES = ["config.json", "training_config.json", "model.safetensors",
               "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
CODE_FILES = ["main.py", "src/__init__.py", "src/training.py", "src/eda.py", "src/preprocessing.py",
              "src/model_source.py", "src/mlm.py", "tools/__init__.py", "tools/analyze_ensemble.py",
              "tools/audit_emoji.py", "tools/audit_precision.py", "tools/cv_oof.py", "tools/refresh_experiments.py",
              "tools/calibrate_fgsm.py", "tools/package_candidate.py", "tools/verify_package.py",
              "requirements.txt", "requirements/train.txt", "requirements/eda.txt", "requirements/lock.txt"]
BENCHMARK_IDS = [str(i) for i in range(1000)]
VAL_IDS = ["a", "b", "c"]
TRAIN_IDS = ["d", "e", "f"]
DATASET = [(i, LABELS[n % 3]) for n, i in enumerate(VAL_IDS + TRAIN_IDS)]
CM = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


class Fixture:
    """Smallest tree that lets refresh.collect() audit single runs and a checkpoint ensemble."""

    def __init__(self, root, n_runs=2):
        self.root = Path(root)
        for name in CODE_FILES:
            self.write_text(name, name)
        self.write_csv("dataset/dataset.csv", "id,text,label", [f"{i},t{i},{label}" for i, label in DATASET])
        self.write_csv("dataset/benchmark.csv", "id,text", [f"{i},t" for i in BENCHMARK_IDS])
        dataset_hash = self.sha("dataset/dataset.csv")
        self.write_json("outputs/eda/split.json", {"dataset_sha256": dataset_hash, "train_ids": TRAIN_IDS,
                                                   "validation_ids": VAL_IDS})
        self.split_hash = self.sha("outputs/eda/split.json")
        self.write_text("report.md", "report")
        self.ids = [f"run{n}" for n in range(n_runs)]
        for n, run_id in enumerate(self.ids):
            ckpt, out = f"saved_models/{run_id}", f"outputs/{run_id}"
            validation = {"confusion_matrix": CM, "macro_f1": 1.0}
            config = {"labels": LABELS, "smoke": False, "dataset_sha256": dataset_hash, "split_sha256": self.split_hash,
                      "train_ids": TRAIN_IDS, "validation_ids": VAL_IDS, "validation": validation,
                      "best_epoch": 3.0, "seed": 42 + n}
            self.write_json(f"{ckpt}/training_config.json", config)
            self.write_json(f"{ckpt}/config.json", {"id2label": {str(i): x for i, x in enumerate(LABELS)}})
            for name in MODEL_FILES[2:]:
                self.write_text(f"{ckpt}/{name}", f"{name}-{n}")
            self.write_json(f"{out}/run_config.json", {"smoke": False, "seed": 42 + n})
            self.write_json(f"{out}/summary.json", {"best_epoch": 3.0, "best_validation_macro_f1": 1.0,
                                                    "reload_verified": True, "reloaded_validation": validation})
        self.predictions = {}
        for n, run_id in enumerate(self.ids):  # custom CSV location, as for the ensemble member CSVs
            path = f"outputs/ens/submissions/model{n}.csv"
            self.write_csv(path, "id,pred", [f"{i},{LABELS[(int(i) + n) % 3]}" for i in BENCHMARK_IDS])
            self.predictions[run_id] = path
        self.write_csv("outputs/ens/submissions/mean.csv", "id,pred", [f"{i},negative" for i in BENCHMARK_IDS])
        self.write_ensemble_summary()
        self.registry = {
            "schema_version": 1, "reviewed_on": "x", "score_provenance": "x",
            "validation_reference": self.ids[0], "comparison_control": None,
            "kaggle_candidate": self.ids[0], "release_candidate": None,
            "runs": [{"id": r, "name": r, "kind": "single", "output": f"outputs/{r}",
                      "checkpoints": [f"saved_models/{r}"], "predictions": self.predictions[r],
                      "report": "report.md", "kaggle_macro_f1": 0.9, "decision": "x"} for r in self.ids],
            "checkpoint_ensembles": [{"id": "ens", "name": "ens", "members": list(self.ids), "output": "outputs/ens",
                                      "report": "report.md", "kaggle_macro_f1": 0.91, "decision": "x"}]}

    def write_ensemble_summary(self):
        models = [{"checkpoint": f"saved_models\\{r}", "validation_macro_f1": 1.0,
                   "benchmark_csv": self.predictions[r].replace("/", "\\"),
                   "benchmark_csv_sha256": self.sha(self.predictions[r])} for r in self.ids]
        self.write_json("outputs/ens/summary.json", {
            "split_sha256": self.split_hash, "labels": LABELS, "validation_rows": 3, "benchmark_rows": 1000,
            "models": models,
            "ensemble": {"method": "mean", "n_models": len(self.ids), "validation_macro_f1": 1.0,
                         "validation_classes": {}, "validation_confusion_matrix": CM,
                         "benchmark_csv": "outputs\\ens\\submissions\\mean.csv",
                         "benchmark_csv_sha256": self.sha("outputs/ens/submissions/mean.csv")}})

    def path(self, relative):
        return self.root / relative

    def write_text(self, relative, text):
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    def write_json(self, relative, data):
        self.write_text(relative, json.dumps(data))

    def write_csv(self, relative, header, lines):
        self.write_text(relative, "\n".join([header] + lines) + "\n")

    def sha(self, relative):
        return hashlib.sha256(self.path(relative).read_bytes()).hexdigest()

    def edit_json(self, relative, change):
        data = json.loads(self.path(relative).read_text(encoding="utf-8"))
        change(data)
        self.write_json(relative, data)

    def collect(self):
        self.write_json("experiments/registry.json", self.registry)
        with mock.patch.object(refresh, "ROOT", self.root.resolve()):
            return refresh.collect()


class RunAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fx = Fixture(self.tmp.name)

    def test_custom_predictions_path_is_used(self):
        snapshot = self.fx.collect()
        run = snapshot["runs"][0]
        self.assertEqual(run["predictions"], "outputs/ens/submissions/model0.csv")
        self.assertEqual(run["predictions_sha256"], self.fx.sha("outputs/ens/submissions/model0.csv"))

    def test_default_predictions_path_is_unchanged(self):
        del self.fx.registry["runs"][1]["predictions"]
        del self.fx.registry["checkpoint_ensembles"]
        self.fx.write_csv("outputs/run1/predictions.csv", "id,pred", [f"{i},neutral" for i in BENCHMARK_IDS])
        run = self.fx.collect()["runs"][1]
        self.assertEqual(run["predictions"], "outputs/run1/predictions.csv")
        self.assertEqual(run["prediction_rows"], 1000)

    def add_init_from(self, sha=None):
        self.fx.write_text("mlm/model.safetensors", "mlm-weights")
        self.fx.write_text("mlm/mlm_summary.json", "{}")
        record = {"path": "mlm", "model_sha256": sha or self.fx.sha("mlm/model.safetensors")}
        for name in ("saved_models/run0/training_config.json", "outputs/run0/run_config.json"):
            self.fx.edit_json(name, lambda d: d.update(init_from=record))

    def test_init_from_checkpoint_is_verified(self):
        self.add_init_from()
        self.assertEqual(len(self.fx.collect()["runs"]), 2)

    def test_init_from_hash_mismatch_rejected(self):
        self.add_init_from(sha="0" * 64)
        with self.assertRaisesRegex(ValueError, "MLM init checkpoint hash differs"):
            self.fx.collect()


class CheckpointEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fx = Fixture(self.tmp.name)

    def test_valid_ensemble_passes_and_renders(self):
        snapshot = self.fx.collect()
        ens = snapshot["checkpoint_ensembles"][0]
        self.assertEqual(ens["csv_sha256"], self.fx.sha("outputs/ens/submissions/mean.csv"))
        self.assertEqual(ens["csv_rows"], 1000)
        text = refresh.render(snapshot)
        self.assertIn("檢查點等權集成", text)
        self.assertIn("0.9100", text)

    def test_no_ensembles_is_empty(self):
        del self.fx.registry["checkpoint_ensembles"]
        self.assertEqual(self.fx.collect()["checkpoint_ensembles"], [])

    def test_ensemble_csv_hash_mismatch_rejected(self):
        self.fx.edit_json("outputs/ens/summary.json",
                          lambda d: d["ensemble"].update(benchmark_csv_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "Ensemble CSV hash differs"):
            self.fx.collect()

    def test_member_csv_hash_mismatch_rejected(self):
        self.fx.edit_json("outputs/ens/summary.json",
                          lambda d: d["models"][0].update(benchmark_csv_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "Member CSV differs"):
            self.fx.collect()

    def test_member_checkpoint_must_match_run(self):
        self.fx.edit_json("outputs/ens/summary.json",
                          lambda d: d["models"][0].update(checkpoint="saved_models\\other"))
        with self.assertRaisesRegex(ValueError, "Ensemble checkpoint differs"):
            self.fx.collect()

    def test_split_hash_must_match(self):
        self.fx.edit_json("outputs/ens/summary.json", lambda d: d.update(split_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self.fx.collect()

    def test_ensemble_csv_ids_must_match_benchmark(self):
        self.fx.write_csv("outputs/ens/submissions/mean.csv", "id,pred", [f"{i},negative" for i in reversed(BENCHMARK_IDS)])
        self.fx.write_ensemble_summary()
        with self.assertRaisesRegex(ValueError, "Ensemble CSV IDs/labels differ"):
            self.fx.collect()

    def test_more_than_three_members_rejected(self):
        self.fx.registry["checkpoint_ensembles"][0]["members"] = self.fx.ids * 2
        with self.assertRaisesRegex(ValueError, "exceeds 3 models"):
            self.fx.collect()
        big = Fixture(self.tmp.name + "/big", n_runs=4)
        with self.assertRaisesRegex(ValueError, "exceeds 3 models"):
            big.collect()


if __name__ == "__main__":
    unittest.main()
