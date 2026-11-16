import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import refresh_experiments as refresh

LABELS = ["negative", "neutral", "positive"]
DATASET_HASH = "d" * 64
MODEL_FILES = ["config.json", "training_config.json", "model.safetensors",
               "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
DATASET = [(str(i), LABELS[i % 3]) for i in range(6)]
FOLDS = [["0", "1", "2"], ["3", "4", "5"]]
BENCHMARK_IDS = [str(i) for i in range(1000)]


def macro_f1(cm):
    f1 = []
    for i in range(3):
        denominator = sum(cm[i]) + sum(row[i] for row in cm)
        f1.append(2 * cm[i][i] / denominator if denominator else 0.0)
    return sum(f1) / 3


def metrics_for(truth, pred):
    cm = [[sum(t == a and p == b for t, p in zip(truth, pred)) for b in range(3)] for a in range(3)]
    return {"confusion_matrix": cm, "macro_f1": macro_f1(cm)}


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.ids = [i for i, _ in DATASET]
        self.truth = dict(DATASET)
        self.write_csv("dataset/dataset.csv", "id,text,label", [f"{i},t{i},{label}" for i, label in DATASET])
        self.manifest = {"k": 2, "dataset_sha256": DATASET_HASH, "folds": []}
        for n, val in enumerate(FOLDS):
            train = [i for i in self.ids if i not in val]
            fold = {"dataset_sha256": DATASET_HASH, "fold": n, "train_ids": train, "validation_ids": val}
            self.write_json(f"folds/fold{n}.json", fold)
            self.manifest["folds"].append({"fold": n, "file": f"fold{n}.json", "sha256": self.sha(f"folds/fold{n}.json")})
            split_hash = self.sha(f"folds/fold{n}.json")
            truth = [LABELS.index(self.truth[i]) for i in val]
            validation = metrics_for(truth, truth)
            config = {"labels": LABELS, "smoke": False, "dataset_sha256": DATASET_HASH, "split_sha256": split_hash,
                      "train_ids": train, "validation_ids": val, "validation": validation, "best_epoch": 3.0, "seed": 42}
            self.write_json(f"ckpt/fold{n}/training_config.json", config)
            self.write_json(f"ckpt/fold{n}/config.json", {"id2label": {str(i): x for i, x in enumerate(LABELS)}})
            for name in MODEL_FILES[2:]:
                self.write_text(f"ckpt/fold{n}/{name}", f"{name}-{n}")
            self.write_json(f"out/fold{n}/run_config.json", {"smoke": False, "seed": 42, "train_ids": train,
                                                           "validation_ids": val, "split_sha256": split_hash})
            self.write_json(f"out/fold{n}/summary.json", {
                "smoke": False, "best_epoch": 3.0, "best_validation_macro_f1": validation["macro_f1"],
                "reload_verified": True, "reloaded_validation": validation, "total_seconds": 10.0})
        self.write_json("folds/folds_manifest.json", self.manifest)
        cm = metrics_for([LABELS.index(label) for _, label in DATASET], [LABELS.index(label) for _, label in DATASET])
        self.write_csv("oof/oof_predictions.csv", "id,label,pred,logit_negative,logit_neutral,logit_positive",
                       [f"{i},{label},{label},0,0,0" for i, label in DATASET])
        self.write_csv("oof/sub.csv", "id,pred", [f"{i},negative" for i in BENCHMARK_IDS])
        self.write_json("oof/oof_metrics.json", {
            "k": 2, "dataset_sha256": DATASET_HASH, "rows": 6, "labels": LABELS,
            "oof": {**cm, "classes": {}}, "folds": {"per_fold": [1.0, 1.0], "mean": 1.0, "sample_std": 0.0},
            "ensemble": {"folds": [0, 1], "sha256": self.sha("oof/sub.csv")}})
        self.entry = {"id": "cv", "name": "fake", "fold_dir": "folds", "report": "report.md",
                      "folds": [{"fold": n, "split": f"folds/fold{n}.json", "output": f"out/fold{n}",
                                 "checkpoint": f"ckpt/fold{n}"} for n in range(2)],
                      "oof_output": "oof",
                      "submission": {"path": "oof/sub.csv", "sha256": self.sha("oof/sub.csv"),
                                     "submission_status": "not_submitted"},
                      "kaggle_macro_f1": None}
        self.write_text("report.md", "report")

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

    def run(self):
        registry = {"cross_validations": [copy.deepcopy(self.entry)]}
        with mock.patch.object(refresh, "ROOT", self.root.resolve()):
            return refresh.collect_cross_validations(registry, DATASET_HASH, BENCHMARK_IDS)


class CrossValidationAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fx = Fixture(self.tmp.name)

    def test_valid_fixture_passes(self):
        result = self.fx.run()
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["folds"]), 2)
        self.assertEqual(result[0]["submission_rows"], 1000)

    def test_no_cross_validations_is_empty(self):
        self.assertEqual(refresh.collect_cross_validations({}, DATASET_HASH, BENCHMARK_IDS), [])

    def test_render_includes_cv_section(self):
        result = self.fx.run()
        snapshot = {"runs": [], "audits": [], "package_rehearsals": [], "calibrations": [], "cross_validations": result,
                    "train_rows": 1, "validation_rows": 1, "confirmation_splits": [],
                    "roles": {"validation_reference": "a", "kaggle_candidate": "b", "comparison_control": None,
                              "release_candidate": None}}
        for cv in snapshot["cross_validations"]:  # fixture has empty per-class dicts
            cv["oof"]["classes"] = {label: {"f1": 1.0} for label in LABELS}
        text = refresh.render(snapshot)
        self.assertIn("交叉驗證", text)
        self.assertIn("not_submitted", text)

    def assert_fails(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            self.fx.run()

    def test_fold_file_differs_from_manifest(self):
        self.fx.edit_json("folds/fold0.json", lambda d: d.update(extra=1))
        self.assert_fails("differs from manifest")

    def test_smoke_checkpoint_rejected(self):
        self.fx.edit_json("ckpt/fold0/training_config.json", lambda d: d.update(smoke=True))
        self.assert_fails("smoke")

    def test_checkpoint_split_ids_must_match_fold(self):
        self.fx.edit_json("ckpt/fold1/training_config.json", lambda d: d.update(validation_ids=["0", "1", "2"]))
        self.assert_fails("Split IDs differ")

    def test_run_config_must_match_training_config(self):
        self.fx.edit_json("out/fold0/run_config.json", lambda d: d.update(seed=7))
        self.assert_fails("Run/checkpoint configs differ")

    def test_reload_verified_required(self):
        self.fx.edit_json("out/fold1/summary.json", lambda d: d.update(reload_verified=False))
        self.assert_fails("reload verification")

    def test_oof_rows_must_cover_dataset(self):
        self.fx.edit_json("oof/oof_metrics.json", lambda d: d.update(rows=5))
        self.assert_fails("OOF metrics provenance")

    def test_oof_metrics_missing(self):
        self.fx.path("oof/oof_metrics.json").unlink()
        with self.assertRaises(OSError):
            self.fx.run()

    def test_submission_hash_must_match(self):
        self.fx.entry["submission"]["sha256"] = "0" * 64
        self.assert_fails("submission hash")

    def test_submission_must_have_1000_rows_in_benchmark_order(self):
        self.fx.write_csv("oof/sub.csv", "id,pred", [f"{i},negative" for i in reversed(BENCHMARK_IDS)])
        self.fx.entry["submission"]["sha256"] = self.fx.sha("oof/sub.csv")
        self.fx.edit_json("oof/oof_metrics.json", lambda d: d["ensemble"].update(sha256=self.fx.sha("oof/sub.csv")))
        self.assert_fails("submission IDs")


if __name__ == "__main__":
    unittest.main()
