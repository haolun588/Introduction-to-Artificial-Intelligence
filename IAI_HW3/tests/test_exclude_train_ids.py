import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from src.training import class_weights, exclude_train_rows, expected_train_ids, train
from tools import cv_oof

LABELS = ("negative", "neutral", "positive")


def make_rows(ids):
    return [{"id": str(i), "label": LABELS[i % 3]} for i in ids]


class ExcludeRowsTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.train, self.val = make_rows(range(9)), make_rows(range(100, 103))

    def listing(self, text):
        path = Path(self.dir.name) / "ex.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def test_removes_only_train_ids_and_records_counts(self):
        path = self.listing("1\n\n4\n100\n")
        kept, record = exclude_train_rows(self.train, self.val, path)
        self.assertEqual([r["id"] for r in kept], ["0", "2", "3", "5", "6", "7", "8"])
        self.assertEqual(len(self.val), 3)
        self.assertEqual({k: v for k, v in record.items() if k not in ("path", "sha256")},
                         {"listed": 3, "removed_from_train": 2, "in_validation_kept": 1,
                          "removed_train_ids": ["1", "4"]})
        self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_duplicate_and_unknown_ids_abort(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            exclude_train_rows(self.train, self.val, self.listing("1\n1\n"))
        with self.assertRaisesRegex(ValueError, "not in the dataset"):
            exclude_train_rows(self.train, self.val, self.listing("1\n999\n"))

    def test_class_weights_follow_filtered_counts(self):
        kept, _ = exclude_train_rows(self.train, self.val, self.listing("0\n"))
        expected = class_weights([r for r in self.train if r["id"] != "0"], "sqrt-inverse")
        self.assertTrue(torch.equal(class_weights(kept, "sqrt-inverse"), expected))
        self.assertFalse(torch.allclose(class_weights(kept, "sqrt-inverse"),
                                        class_weights(self.train, "sqrt-inverse")))


class CvOofCheckTests(unittest.TestCase):
    fold = {"train_ids": ["1", "2", "3"], "validation_ids": ["9"]}

    def test_no_record_is_strict_as_before(self):
        cv_oof.check_fold_split({"train_ids": ["1", "2", "3"], "validation_ids": ["9"]}, self.fold)
        with self.assertRaises(ValueError):
            cv_oof.check_fold_split({"train_ids": ["1", "3"], "validation_ids": ["9"]}, self.fold)

    def test_recorded_exclusion_is_accepted_and_still_strict(self):
        record = {"removed_train_ids": ["2"], "removed_from_train": 1}
        cv_oof.check_fold_split({"train_ids": ["1", "3"], "validation_ids": ["9"],
                                 "exclude_train_ids": record}, self.fold)
        for bad in ({"train_ids": ["1"], "validation_ids": ["9"], "exclude_train_ids": record},
                    {"train_ids": ["1", "2", "3"], "validation_ids": ["9"], "exclude_train_ids": record},
                    {"train_ids": ["1", "3"], "validation_ids": ["8"], "exclude_train_ids": record}):
            with self.assertRaises(ValueError):
                cv_oof.check_fold_split(bad, self.fold)
        with self.assertRaises(ValueError):
            expected_train_ids(["1", "3"], {"exclude_train_ids": {"removed_train_ids": ["7"], "removed_from_train": 1}})


class TrainIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_train(self, exclude_file, set_attribute=True):
        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 3)

            def forward(self, input_ids):
                return SimpleNamespace(logits=self.linear(input_ids))

            def save_pretrained(self, path, **kwargs):
                torch.save(self.state_dict(), path / "fixture.pt")

        tokenizer = Mock()
        train_rows, val_rows = make_rows(range(12)), make_rows(range(100, 106))
        seen = {"train": [], "other": []}

        def batches(rows, tokenizer, max_length, batch_size, shuffle=False, seed=42, **kwargs):
            seen["train" if shuffle else "other"].append([r["id"] for r in rows])
            data = [{"input_ids": torch.tensor([i / 10, 1.0]), "labels": LABELS.index(r["label"])}
                    for i, r in enumerate(rows)]
            return torch.utils.data.DataLoader(data, batch_size=batch_size)

        def reload_model(path, device):
            model = Classifier()
            model.load_state_dict(torch.load(path / "fixture.pt", weights_only=True))
            return model.eval(), tokenizer, json.loads((path / "training_config.json").read_text())

        root = Path(self.tmp.name)
        split = root / "split.json"
        split.write_text("{}")
        args = SimpleNamespace(epochs=1, eval_interval=1.0, batch_size=2, accumulation_steps=1, max_length=64,
                               learning_rate=0.01, warmup_ratio=0.1, smoke=False, output_dir=root / "run",
                               checkpoint=root / "ckpt", base_model=root / "base", device="cpu", no_amp=True,
                               seed=42, data_dir=root, split=split, class_weighting="sqrt-inverse",
                               no_gradient_checkpointing=True, preprocessing="whitespace-v1")
        if set_attribute:
            args.exclude_train_ids = exclude_file
        with patch("src.training.verify_base_model", return_value={"directory": "x", "model": "f", "revision": "f"}), \
                patch("src.training.AutoModelForSequenceClassification.from_pretrained",
                      side_effect=lambda *a, **k: Classifier()), \
                patch("src.training.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                patch("src.training.load_split", return_value=(train_rows, val_rows, {"dataset_sha256": "f"})), \
                patch("src.training.loader", side_effect=batches), patch("src.training.plot_history"), \
                patch("src.training.load_checkpoint", side_effect=reload_model), \
                redirect_stdout(io.StringIO()):
            train(args)
        run = json.loads((args.output_dir / "run_config.json").read_text())
        saved = json.loads((args.checkpoint / "training_config.json").read_text())
        return run, saved, seen

    def test_absent_option_matches_legacy_behaviour(self):
        for none_value, set_attribute in ((None, True), (None, False)):
            self.tmp.cleanup()
            self.tmp = tempfile.TemporaryDirectory()
            run, saved, _ = self.run_train(none_value, set_attribute)
            self.assertIsNone(run["exclude_train_ids"])
            self.assertIsNone(saved["exclude_train_ids"])
            self.assertEqual(run["train_ids"], [str(i) for i in range(12)])
            self.assertEqual(run["class_weights"], class_weights(make_rows(range(12)), "sqrt-inverse").tolist())

    def test_exclusion_end_to_end(self):
        listing = Path(self.tmp.name) / "ex.txt"
        listing.write_text("0\n1\n100\n", encoding="utf-8")
        run, saved, seen = self.run_train(listing)
        self.assertEqual(run["train_ids"], [str(i) for i in range(2, 12)])
        self.assertEqual(run["validation_ids"], [str(i) for i in range(100, 106)])
        record = run["exclude_train_ids"]
        self.assertEqual((record["listed"], record["removed_from_train"], record["in_validation_kept"]), (3, 2, 1))
        self.assertEqual(saved["exclude_train_ids"], record)
        self.assertEqual(seen["train"][0], run["train_ids"])
        self.assertEqual(seen["other"][-1], run["validation_ids"])
        self.assertEqual(run["class_weights"], class_weights(make_rows(range(2, 12)), "sqrt-inverse").tolist())
        cv_oof.check_fold_split(saved, {"train_ids": [str(i) for i in range(12)],
                                        "validation_ids": run["validation_ids"]})

    def test_bad_list_aborts(self):
        listing = Path(self.tmp.name) / "ex.txt"
        listing.write_text("1\n1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_train(listing)
        listing.write_text("424242\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not in the dataset"):
            self.run_train(listing)


if __name__ == "__main__":
    unittest.main()
