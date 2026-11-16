import csv
import unittest
from pathlib import Path

from src.eda import LABELS, grouped_kfold, grouped_split, preprocess


def _rows():
    rows, n = [], 0
    for label, sizes in (("negative", [5, 3, 3, 2, 2] + [1] * 20), ("neutral", [4, 2] + [1] * 14),
                         ("positive", [3, 2, 2] + [1] * 12)):
        for g, size in enumerate(sizes):
            for _ in range(size):
                rows.append({"id": str(n), "text": f"{label}  text {g}", "label": label})
                n += 1
    return rows


class KFoldTests(unittest.TestCase):
    def test_partition_groups_and_balance(self):
        rows = _rows()
        folds = grouped_kfold(rows, 5, 42)
        all_ids = [i for f in folds for i in f]
        self.assertEqual(sorted(all_ids, key=int), [r["id"] for r in rows])
        self.assertEqual(len(all_ids), len(set(all_ids)))
        by_id = {r["id"]: r for r in rows}
        homes = {}
        for n, fold in enumerate(folds):
            self.assertEqual({by_id[i]["label"] for i in fold}, set(LABELS))
            for i in fold:
                self.assertEqual(homes.setdefault(preprocess(by_id[i]["text"]), n), n)
        for label in LABELS:
            counts = [sum(by_id[i]["label"] == label for i in f) for f in folds]
            sizes = {}
            for r in rows:
                if r["label"] == label:
                    sizes[r["text"]] = sizes.get(r["text"], 0) + 1
            self.assertLessEqual(max(counts) - min(counts), max(sizes.values()))

    def test_actual_data(self):
        with (Path(__file__).resolve().parents[1] / "dataset/dataset.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        folds = grouped_kfold(rows, 5, 42)
        self.assertEqual(sum(map(len, folds)), len(rows))
        self.assertEqual(len(set().union(*folds)), len(rows))

    def test_reproducible_and_seed_sensitive(self):
        rows = _rows()
        self.assertEqual(grouped_kfold(rows, 5, 42), grouped_kfold(rows, 5, 42))
        self.assertNotEqual(grouped_kfold(rows, 5, 42), grouped_kfold(rows, 5, 43))

    def test_conflict_and_small_k(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            grouped_kfold([{'id': '1', 'text': 'a  b', 'label': 'positive'},
                           {'id': '2', 'text': 'a b', 'label': 'negative'}], 2)
        for k in (0, 1, -3):
            with self.assertRaises(ValueError):
                grouped_kfold(_rows(), k)


class SplitTests(unittest.TestCase):
    def test_actual_data_partition_and_reproducibility(self):
        with (Path(__file__).resolve().parents[1] / "dataset/dataset.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        train, validation = grouped_split(rows)
        self.assertEqual((len(train), len(validation)), (8100, 900))
        self.assertEqual(grouped_split(rows), (train, validation))
        reversed_train, reversed_validation = grouped_split(list(reversed(rows)))
        self.assertEqual({r['id'] for r in validation}, {r['id'] for r in reversed_validation})
        self.assertFalse({preprocess(r['text']) for r in train} & {preprocess(r['text']) for r in validation})
        self.assertFalse({r['id'] for r in train} & {r['id'] for r in validation})
        self.assertEqual({r['id'] for r in train + validation}, {r['id'] for r in rows})
        self.assertNotEqual(grouped_split(rows, seed=43), (train, validation))

    def test_conflicting_normalized_duplicate(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            grouped_split([{'id': '1', 'text': 'a  b', 'label': 'positive'},
                           {'id': '2', 'text': 'a b', 'label': 'negative'}])

    def test_invalid_fraction(self):
        for fraction in (0, 1, -0.1):
            with self.assertRaises(ValueError):
                grouped_split([], validation_fraction=fraction)


if __name__ == '__main__':
    unittest.main()
