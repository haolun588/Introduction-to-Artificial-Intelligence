import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training import LABELS
from tools import ensemble_checkpoints as ec
from tools.cv_oof import softmax


class CheckArgsTests(unittest.TestCase):
    def test_more_than_three_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "At most 3"):
                ec.check_args(["a", "b", "c", "d"], Path(tmp) / "out")

    def test_empty_list_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                ec.check_args([], Path(tmp) / "out")
            with self.assertRaisesRegex(ValueError, "distinct"):
                ec.check_args(["a", "a"], Path(tmp) / "out")

    def test_nonempty_output_dir_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "x.txt").write_text("x")
            with self.assertRaisesRegex(ValueError, "not empty"):
                ec.check_args(["a"], Path(tmp))
            ec.check_args(["a"], Path(tmp) / "new")  # missing dir is fine

    def test_split_and_dataset_hash_mismatch_rejected(self):
        config = {"split_sha256": "s", "dataset_sha256": "d", "validation_ids": ["1"]}
        ec.check_checkpoint_config(config, "s", "d", ["1"])
        with self.assertRaisesRegex(ValueError, "split_sha256"):
            ec.check_checkpoint_config(config, "other", "d", ["1"])
        with self.assertRaisesRegex(ValueError, "dataset_sha256"):
            ec.check_checkpoint_config(config, "s", "other", ["1"])
        with self.assertRaisesRegex(ValueError, "validation ids"):
            ec.check_checkpoint_config(config, "s", "d", ["2"])


class BuildReportTests(unittest.TestCase):
    def test_ensemble_mean_csv_and_hashes(self):
        rng = np.random.default_rng(1)
        truth = [0, 1, 2, 0, 1, 2]
        val = [rng.normal(size=(6, 3)) for _ in range(3)]
        bench = [rng.normal(size=(5, 3)) for _ in range(3)]
        ids = [f"b{i}" for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            split = tmp / "split.json"
            split.write_text("{}")
            report = ec.build_report(tmp / "out", ["c0", "c1", "c2"], truth, val, ids, bench, split)
            saved = json.loads((tmp / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["ensemble"]["benchmark_csv_sha256"], report["ensemble"]["benchmark_csv_sha256"])
            expected = np.mean([softmax(b) for b in bench], axis=0).argmax(axis=1)
            ens_csv = Path(report["ensemble"]["benchmark_csv"])
            with ens_csv.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], ["id", "pred"])
            self.assertEqual(rows[1:], [[i, LABELS[p]] for i, p in zip(ids, expected)])
            self.assertEqual(report["ensemble"]["benchmark_csv_sha256"], hashlib.sha256(ens_csv.read_bytes()).hexdigest())
            for n, model in enumerate(report["models"]):
                path = Path(model["benchmark_csv"])
                self.assertEqual(model["benchmark_csv_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                with path.open(encoding="utf-8", newline="") as stream:
                    body = list(csv.reader(stream))[1:]
                self.assertEqual(body, [[i, LABELS[p]] for i, p in zip(ids, bench[n].argmax(axis=1))])
            self.assertEqual(report["split_sha256"], hashlib.sha256(b"{}").hexdigest())
            self.assertEqual(len(report["models"]), 3)
            self.assertIn("validation_confusion_matrix", report["ensemble"])

    def test_ensemble_validation_f1_uses_softmax_mean(self):
        truth = [0, 1]
        # model A confidently says [0,0]; model B weakly says [1,1]; softmax mean follows the confident model
        a = np.array([[10.0, 0, 0], [10.0, 0, 0]])
        b = np.array([[0, 1.0, 0], [0, 1.0, 0]])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            split = tmp / "s.json"
            split.write_text("{}")
            report = ec.build_report(tmp / "o", ["a", "b"], truth, [a, b], ["x"], [a[:1], b[:1]], split)
        expected = np.mean([softmax(a), softmax(b)], axis=0)
        self.assertEqual(expected.argmax(axis=1).tolist(), [0, 0])
        from tools.cv_oof import macro_summary
        self.assertEqual(report["ensemble"]["validation_macro_f1"], macro_summary(truth, expected)["macro_f1"])
        self.assertEqual(report["ensemble"]["validation_confusion_matrix"],
                         macro_summary(truth, expected)["confusion_matrix"])


class MainTests(unittest.TestCase):
    def test_main_rejects_four_checkpoints_before_inference(self):
        called = []
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                ec.main(["--checkpoints", "a", "b", "c", "d", "--output-dir", str(Path(tmp) / "o")],
                        infer=lambda *a: called.append(a))
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
