import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.training import metrics
from tools import compare_oof


def make_side(ids, truth, pred, per_fold=None, seed=42):
    return {"metrics": {"dataset_sha256": "x", "k": 2, "seed": seed, "folds": {"per_fold": per_fold}},
            "ids": list(ids), "truth": np.array(truth), "pred": np.array(pred)}


class CompareOofTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.n = 200
        self.ids = [str(i) for i in range(self.n)]
        self.truth = rng.integers(0, 3, self.n)
        self.pred_a = np.where(rng.random(self.n) < 0.8, self.truth, rng.integers(0, 3, self.n))
        self.pred_b = np.where(rng.random(self.n) < 0.85, self.truth, rng.integers(0, 3, self.n))
        self.fold_ids = [self.ids[:100], self.ids[100:]]

    def side(self, pred, **kwargs):
        positions = compare_oof.fold_positions(self.ids, self.fold_ids)
        scores = compare_oof.fold_scores(self.truth, pred, positions)
        return make_side(self.ids, self.truth, pred, scores, **kwargs)

    def test_rejects_id_or_label_mismatch(self):
        a = self.side(self.pred_a)
        swapped = self.side(self.pred_b)
        swapped["ids"] = swapped["ids"][::-1]
        with self.assertRaisesRegex(ValueError, "id order"):
            compare_oof.check_compatible(a, swapped)
        other = self.side(self.pred_b)
        other["truth"] = (other["truth"] + 1) % 3
        with self.assertRaisesRegex(ValueError, "labels"):
            compare_oof.check_compatible(a, other)
        with self.assertRaisesRegex(ValueError, "seed"):
            compare_oof.check_compatible(a, self.side(self.pred_b, seed=7))

    def test_rejects_fold_scores_not_matching_split(self):
        a = self.side(self.pred_a)
        b = self.side(self.pred_b)
        b["metrics"]["folds"]["per_fold"] = [0.1, 0.2]
        with self.assertRaisesRegex(ValueError, "per-fold"):
            compare_oof.compare(a, b, self.fold_ids, resamples=5)

    def test_identical_predictions_give_zero_delta(self):
        report = compare_oof.compare(self.side(self.pred_a), self.side(self.pred_a), self.fold_ids, resamples=50)
        self.assertEqual(report["delta_oof_macro_f1"], 0)
        boot = report["paired_bootstrap"]
        self.assertEqual((boot["delta_p2_5"], boot["delta_p97_5"], boot["fraction_delta_le_0"]), (0, 0, 1))
        self.assertEqual(report["disagreement"]["rows"], 0)
        self.assertEqual(report["per_fold"]["b_wins"], 0)

    def test_vectorized_macro_f1_matches_training_metrics(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            t, p = rng.integers(0, 3, 50), rng.integers(0, 3, 50)
            self.assertAlmostEqual(compare_oof.macro_f1(t, p), metrics(t.tolist(), p.tolist())["macro_f1"], places=12)
        t = np.array([0, 0, 1, 1])  # class 2 absent from both truth and prediction
        self.assertAlmostEqual(compare_oof.macro_f1(t, t), metrics(t.tolist(), t.tolist())["macro_f1"], places=12)

    def test_bootstrap_reproducible_and_seed_sensitive(self):
        args = (self.truth, self.pred_a, self.pred_b)
        first = compare_oof.paired_bootstrap(*args, seed=5, resamples=100)
        self.assertEqual(first, compare_oof.paired_bootstrap(*args, seed=5, resamples=100))
        self.assertNotEqual(first, compare_oof.paired_bootstrap(*args, seed=6, resamples=100))

    def test_compare_reports_disagreement_and_fold_wins(self):
        report = compare_oof.compare(self.side(self.pred_a), self.side(self.pred_b), self.fold_ids, resamples=50)
        differ = self.pred_a != self.pred_b
        self.assertEqual(report["disagreement"]["rows"], int(differ.sum()))
        self.assertEqual(report["disagreement"]["a_correct"], int((differ & (self.pred_a == self.truth)).sum()))
        self.assertAlmostEqual(report["delta_oof_macro_f1"],
                               metrics(self.truth.tolist(), self.pred_b.tolist())["macro_f1"]
                               - metrics(self.truth.tolist(), self.pred_a.tolist())["macro_f1"], places=12)

    def test_main_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            out.write_text("{}", encoding="utf-8")
            argv = ["compare_oof", "--a", "a", "--b", "b", "--output", str(out)]
            with mock.patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    compare_oof.main()
            self.assertEqual(out.read_text(encoding="utf-8"), "{}")


if __name__ == "__main__":
    unittest.main()
