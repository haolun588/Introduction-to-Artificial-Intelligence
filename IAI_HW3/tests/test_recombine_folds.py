import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training import LABELS
from tools import cv_oof, recombine_folds


def write_npz(directory, fold, ids, logits):
    np.savez(Path(directory) / f"logits_benchmark_fold{fold}.npz", ids=np.array(ids), logits=np.asarray(logits))


class RecombineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        rng = np.random.default_rng(0)
        self.ids = [f"id{i}" for i in range(20)]
        self.logits = [rng.normal(size=(20, 3)).astype(np.float32) * 3 for _ in range(5)]
        for f in range(5):
            write_npz(self.dir, f, self.ids, self.logits[f])

    def test_equal_weight_softmax_mean(self):
        out = self.dir / "out.csv"
        recombine_folds.main(["--oof-dir", str(self.dir), "--folds", "0", "2", "4", "--output", str(out)])
        probs = np.mean([cv_oof.softmax(self.logits[f]) for f in (0, 2, 4)], axis=0)
        with out.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["id", "pred"])
        self.assertEqual(rows[1:], [[i, LABELS[p]] for i, p in zip(self.ids, probs.argmax(1))])
        # byte-identical to cv_oof's own writer / ensemble for up to 3 folds
        ref = self.dir / "ref.csv"
        cv_oof.write_pred_csv(ref, self.ids, cv_oof.ensemble_predictions(self.logits, [0, 2, 4]))
        self.assertEqual(out.read_bytes(), ref.read_bytes())

    def test_five_folds_and_reference_disagreement(self):
        ref = self.dir / "ref.csv"
        recombine_folds.main(["--oof-dir", str(self.dir), "--folds", "0", "--output", str(ref)])
        out = self.dir / "out5.csv"
        recombine_folds.main(["--oof-dir", str(self.dir), "--folds", "0", "1", "2", "3", "4",
                              "--output", str(out), "--reference", str(ref)])
        ref_ids, ref_pred = recombine_folds.read_pred_csv(ref)
        new_ids, new_pred = recombine_folds.read_pred_csv(out)
        self.assertEqual(ref_ids, new_ids)
        n, trans = recombine_folds.disagreement(
            [LABELS.index(p) for p in ref_pred], [LABELS.index(p) for p in new_pred])
        self.assertEqual(n, sum(a != b for a, b in zip(ref_pred, new_pred)))
        self.assertEqual(sum(trans.values()), n)

    def test_ids_mismatch_raises(self):
        write_npz(self.dir, 3, list(reversed(self.ids)), self.logits[3])
        with self.assertRaisesRegex(ValueError, "ids"):
            recombine_folds.main(["--oof-dir", str(self.dir), "--folds", "0", "3",
                                  "--output", str(self.dir / "o.csv")])

    def test_bad_logits_raise(self):
        write_npz(self.dir, 1, self.ids, self.logits[1][:, :2])
        with self.assertRaisesRegex(ValueError, "shape"):
            recombine_folds.load_fold_logits(self.dir, [1])
        bad = self.logits[2].copy()
        bad[0, 0] = np.nan
        write_npz(self.dir, 2, self.ids, bad)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            recombine_folds.load_fold_logits(self.dir, [2])

    def test_duplicate_or_out_of_range_folds_raise(self):
        for folds in (["0", "0"], ["5"], ["-1"], ["0", "1", "2", "3", "4", "4"]):
            with self.assertRaises(ValueError):
                recombine_folds.main(["--oof-dir", str(self.dir), "--folds", *folds,
                                      "--output", str(self.dir / "o.csv")])
        self.assertFalse((self.dir / "o.csv").exists())

    def test_existing_output_raises(self):
        out = self.dir / "exists.csv"
        out.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            recombine_folds.main(["--oof-dir", str(self.dir), "--folds", "0", "--output", str(out)])
        self.assertEqual(out.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
