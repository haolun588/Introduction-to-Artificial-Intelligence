import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training import LABELS, metrics
from tools import cv_oof


class OofTests(unittest.TestCase):
    def test_assemble_rejects_duplicate_and_missing(self):
        logits = np.zeros((2, 3))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            cv_oof.assemble_oof([(["a", "b"], logits), (["b"], logits[:1])], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "missing"):
            cv_oof.assemble_oof([(["a", "b"], logits)], ["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "unexpected"):
            cv_oof.assemble_oof([(["a", "b"], logits)], ["a"])

    def test_assemble_orders_by_all_ids(self):
        out = cv_oof.assemble_oof([(["b"], [[0, 1, 2]]), (["a"], [[3, 4, 5]])], ["a", "b"])
        self.assertEqual(out.tolist(), [[3, 4, 5], [0, 1, 2]])

    def test_metrics_match_training_metrics(self):
        rng = np.random.default_rng(0)
        logits, truth = rng.normal(size=(60, 3)), rng.integers(0, 3, 60)
        expected = metrics(truth.tolist(), logits.argmax(1).tolist())
        self.assertEqual(cv_oof.macro_summary(truth, logits), expected)

    def test_fold_statistics(self):
        s = cv_oof.fold_statistics([0.5, 0.7])
        self.assertAlmostEqual(s["mean"], 0.6)
        self.assertAlmostEqual(s["sample_std"], 0.1414213562, places=6)

    def test_ensemble_limit_and_validity(self):
        bench = [np.zeros((4, 3)) for _ in range(5)]
        with self.assertRaises(ValueError):
            cv_oof.ensemble_predictions(bench, [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            cv_oof.ensemble_predictions(bench, [0, 0])
        with self.assertRaises(ValueError):
            cv_oof.ensemble_predictions(bench, [7])

    def test_ensemble_softmax_mean(self):
        a = np.array([[10.0, 0, 0], [0, 0, 5.0]])
        b = np.array([[0, 3.0, 0], [0, 0, 5.0]])
        self.assertEqual(cv_oof.ensemble_predictions([a, b], [0, 1]).tolist(), [0, 2])

    def test_csv_order_and_format(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.csv"
            cv_oof.write_pred_csv(path, ["5", "2", "9"], np.array([0, 1, 2]))
            with path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
        self.assertEqual(rows, [["id", "pred"], ["5", "negative"], ["2", "neutral"], ["9", "positive"]])
        self.assertEqual(LABELS, ("negative", "neutral", "positive"))

    def test_fold_split_mismatch_fails(self):
        fold = {"train_ids": ["1", "2"], "validation_ids": ["3"]}
        cv_oof.check_fold_split({"train_ids": ["1", "2"], "validation_ids": ["3"]}, fold)
        for bad in ({"train_ids": ["1"], "validation_ids": ["3"]},
                    {"train_ids": ["1", "2"], "validation_ids": ["2"]}, {}):
            with self.assertRaises(ValueError):
                cv_oof.check_fold_split(bad, fold)


if __name__ == "__main__":
    unittest.main()
