"""Check that the audit distinguishes probability drift from changed decisions."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit_precision import compare_outputs, save_records


class PrecisionAuditTests(unittest.TestCase):
    def test_probability_drift_and_label_flip_are_separate(self):
        amp = np.array([[0.6, 0.3, 0.1], [0.49, 0.50, 0.01]])
        fp32 = np.array([[0.7, 0.2, 0.1], [0.51, 0.48, 0.01]])
        result = compare_outputs(amp, fp32, amp.argmax(1), fp32.argmax(1), ["first", "boundary"])
        self.assertEqual(result["disagreement_ids"], ["boundary"])
        self.assertEqual(result["label_disagreements"], 1)
        self.assertAlmostEqual(result["probability_abs_diff_max"], 0.1)
        # A changed probability distribution alone must not count as a changed label.
        stable = compare_outputs(amp[:1], fp32[:1], amp[:1].argmax(1), fp32[:1].argmax(1), ["first"])
        self.assertEqual(stable["label_disagreements"], 0)

    def test_invalid_probabilities_fail_closed(self):
        valid = np.array([[0.6, 0.3, 0.1]])
        with self.assertRaises(ValueError):
            compare_outputs(valid, np.array([[np.nan, 0, 1]]), np.array([0]), np.array([2]), ["id"])
        with self.assertRaises(AssertionError):
            compare_outputs(valid, valid * 2, np.array([0]), np.array([0]), ["id"])

    def test_export_keeps_ids_order_labels_and_header_for_empty_disagreements(self):
        rows = [{"id": "20"}, {"id": "3"}]
        probs = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]])
        values = {mode: (probs, np.array([2, 0])) for mode in ("amp", "fp32")}
        with tempfile.TemporaryDirectory() as directory:
            save_records(Path(directory), rows, values)
            with (Path(directory) / "predictions_fp32.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [{"id": "20", "pred": "positive"}, {"id": "3", "pred": "negative"}])
            with (Path(directory) / "disagreements.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertIn("amp_pred", reader.fieldnames)
                self.assertEqual(list(reader), [])


if __name__ == "__main__":
    unittest.main()
