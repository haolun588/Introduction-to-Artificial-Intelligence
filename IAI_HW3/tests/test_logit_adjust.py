"""Unit tests for post-hoc logit adjustment (no GPU or model needed)."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training import LABELS
from tools.logit_adjust import (TAU_GRID, adjust_logits, check_pred_csv, class_prior, flat_taus,
                                predict_adjusted, read_pred_csv, select_tau, write_pred_csv)


class LogitAdjustTests(unittest.TestCase):
    def setUp(self):
        _, self.prior = class_prior(["negative"] * 6 + ["neutral"] * 2 + ["positive"] * 2)
        rng = np.random.default_rng(0)
        self.logits = rng.normal(size=(500, 3)) + np.array([1.0, 0.0, 0.0])

    def test_tau_zero_keeps_argmax(self):
        np.testing.assert_array_equal(predict_adjusted(self.logits, self.prior, 0.0), self.logits.argmax(1))
        np.testing.assert_array_equal(adjust_logits(self.logits, self.prior, 0.0), self.logits)

    def test_positive_tau_adds_minority_predictions(self):
        base = np.count_nonzero(predict_adjusted(self.logits, self.prior, 0.0) != 0)
        shifted = np.count_nonzero(predict_adjusted(self.logits, self.prior, 1.0) != 0)
        reverse = np.count_nonzero(predict_adjusted(self.logits, self.prior, -1.0) != 0)
        self.assertGreater(shifted, base)
        self.assertLess(reverse, base)

    def test_grid_is_fixed(self):
        self.assertEqual(len(TAU_GRID), 21)
        self.assertEqual(TAU_GRID[0], -1.0)
        self.assertEqual(TAU_GRID[-1], 1.0)
        self.assertIn(0.0, TAU_GRID)
        np.testing.assert_allclose(np.diff(TAU_GRID), 0.1, atol=1e-12)

    def test_select_tau_tie_rules(self):
        taus = [-0.2, -0.1, 0.0, 0.1, 0.2]
        self.assertEqual(select_tau(taus, [0.9, 0.9, 0.8, 0.9, 0.9]), -0.1)
        self.assertEqual(select_tau(taus, [0.9, 0.8, 0.8, 0.8, 0.9]), -0.2)  # equal |tau| -> smaller tau
        self.assertEqual(select_tau(taus, [0.8, 0.8, 0.9, 0.8, 0.8]), 0.0)
        self.assertEqual(select_tau(taus, [0.8, 0.8, 0.7, 0.85, 0.8]), 0.1)

    def test_flat_range(self):
        taus = [-0.1, 0.0, 0.1, 0.2]
        self.assertEqual(flat_taus(taus, [0.90, 0.91, 0.909, 0.95]), [0.2])
        self.assertEqual(flat_taus(taus, [0.949, 0.90, 0.95, 0.9]), [-0.1, 0.1])

    def test_csv_format(self):
        ids = [f"id{i}" for i in range(1000)]
        pred = np.arange(1000) % 3
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.csv"
            write_pred_csv(path, ids, pred)
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], ["id", "pred"])
            self.assertEqual(len(rows), 1001)
            self.assertTrue(all(r[1] in LABELS for r in rows[1:]))
            check_pred_csv(path, ids)
            read_ids, read_pred = read_pred_csv(path)
            self.assertEqual(read_ids, ids)
            np.testing.assert_array_equal(read_pred, pred)
            with self.assertRaises(ValueError):
                check_pred_csv(path, ids[::-1])
            path.write_text("id,pred\na,bogus\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_pred_csv(path)


if __name__ == "__main__":
    unittest.main()
