import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.training import LABELS
from tools import audit_label_noise as aln


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    oof, data = [], []
    for i in range(n):
        label = LABELS[int(rng.integers(0, 3))]
        pred = label if rng.random() < 0.6 else LABELS[int(rng.integers(0, 3))]
        logits = rng.normal(0, 1, 3)
        logits[LABELS.index(pred)] += rng.uniform(0, 6)
        row = {"id": str(i), "label": label, "pred": pred}
        row.update({f"logit_{l}": str(float(v)) for l, v in zip(LABELS, logits)})
        oof.append(row)
        data.append({"id": str(i), "text": f"text {i}", "label": label})
    return oof, data


class AuditLabelNoiseTests(unittest.TestCase):
    def test_softmax_and_p_pred(self):
        p = aln.softmax([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
        np.testing.assert_allclose(p[0], [1 / 3] * 3)
        np.testing.assert_allclose(p[1], [1, 0, 0], atol=1e-12)
        oof = [{"id": "0", "label": "negative", "pred": "positive", "logit_negative": "0",
                "logit_neutral": "0", "logit_positive": str(np.log(99))}]
        data = [{"id": "0", "text": "t", "label": "negative"}]
        _, _, _, cand = aln.analyze(oof, data)
        self.assertAlmostEqual(cand[0]["p_pred"], 99 / 101)
        self.assertAlmostEqual(cand[0]["p_label"], 1 / 101)

    def test_threshold_counts(self):
        oof, data = make_data()
        stats, *_ = aln.analyze(oof, data)
        probs = aln.softmax([[float(r[f"logit_{l}"]) for l in LABELS] for r in oof])
        for t in aln.THRESHOLDS:
            expected = sum(1 for r, p in zip(oof, probs)
                           if r["pred"] != r["label"] and p[LABELS.index(r["pred"])] >= t)
            got = stats["pred_prob_thresholds"][f"{t:.2f}"]
            self.assertEqual(got["total"], expected)
            self.assertEqual(sum(got["by_label_pred"].values()), expected)
        self.assertEqual(sum(stats["class_counts"].values()), len(oof))

    def test_group_a_sorted_and_capped(self):
        oof, data = make_data(n=2000)
        stats, key, _, cand = aln.analyze(oof, data)
        a = sorted((r for r in key if r["group"] == "A"), key=lambda r: -r["p_pred"])
        expected = [r for r in cand if r["label"] == "neutral"][:aln.A_LIMIT]
        self.assertEqual(len(a), aln.A_LIMIT)
        self.assertEqual([r["id"] for r in a], [r["id"] for r in expected])
        self.assertEqual(stats["group_sizes"], {"A": 40, "B": 20, "C": 20})
        self.assertEqual([c["p_pred"] for c in cand], sorted((c["p_pred"] for c in cand), reverse=True))

    def test_small_pools_take_all(self):
        oof, data = make_data(n=30)
        stats, *_ = aln.analyze(oof, data)
        self.assertLessEqual(stats["group_sizes"]["A"], aln.A_LIMIT)
        self.assertLessEqual(stats["group_sizes"]["C"], aln.C_LIMIT)

    def test_reproducible_and_seed_sensitive(self):
        oof, data = make_data(n=2000)
        first = aln.analyze(oof, data, seed=5)
        self.assertEqual(first, aln.analyze(oof, data, seed=5))
        self.assertNotEqual(first[1], aln.analyze(oof, data, seed=6)[1])

    def test_blind_does_not_leak(self):
        oof, data = make_data(n=2000)
        _, key, blind, _ = aln.analyze(oof, data)
        self.assertEqual(len(blind), len(key))
        for row in blind:
            self.assertEqual(set(row), {"review_id", "text"})
        self.assertEqual([r["review_id"] for r in blind], [f"R{i:03d}" for i in range(1, len(blind) + 1)])

    def test_rejects_id_or_label_mismatch(self):
        oof, data = make_data(n=20)
        with self.assertRaisesRegex(ValueError, "ids differ"):
            aln.analyze(oof[:-1], data)
        bad = [dict(r) for r in data]
        bad[3]["label"] = "neutral" if bad[3]["label"] != "neutral" else "positive"
        with self.assertRaisesRegex(ValueError, "Label mismatch"):
            aln.analyze(oof, bad)

    def test_main_refuses_nonempty_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "x.txt").write_text("keep", encoding="utf-8")
            with mock.patch("sys.argv", ["audit", "--output-dir", tmp]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    aln.main()
            self.assertEqual((Path(tmp) / "x.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
