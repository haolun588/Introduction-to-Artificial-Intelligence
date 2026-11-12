import csv
import unittest
from pathlib import Path

from eda import grouped_split, preprocess


class SplitTests(unittest.TestCase):
    def test_actual_data_partition_and_reproducibility(self):
        with (Path(__file__).parent / "dataset/dataset.csv").open(encoding="utf-8-sig", newline="") as f:
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
