"""Source checks use small fixtures and a tiny CPU module; no BERT training."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from src import model_source
from src.model_source import verify_base_model
from src.training import train


class ModelSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "copied-approved-snapshot"
        self.base.mkdir()
        self.contents = {
            name: (json.dumps({"_name_or_path": model_source.MODEL}).encode()
                   if name == "config.json" else ("fixture " + name).encode())
            for name in model_source.APPROVED_FILES
        }
        self.expected = {name: (len(data), hashlib.sha256(data).hexdigest())
                         for name, data in self.contents.items()}
        for name, data in self.contents.items():
            (self.base / name).write_bytes(data)
        pinned = patch.object(model_source, "APPROVED_FILES", self.expected)
        pinned.start()
        self.addCleanup(pinned.stop)

    def test_valid_copy_records_actual_path_and_every_file_hash(self):
        (self.base / ".cache").mkdir()
        result = verify_base_model(self.base)
        self.assertEqual(result["directory"], str(self.base.resolve()))
        self.assertEqual(result["model"], model_source.MODEL)
        self.assertEqual(result["revision"], model_source.REVISION)
        self.assertEqual(set(result["files"]), set(self.contents))
        for name, content in self.contents.items():
            self.assertEqual(result["files"][name]["sha256"], hashlib.sha256(content).hexdigest())

    def test_changed_bytes_are_rejected_for_each_required_file(self):
        for name, content in self.contents.items():
            with self.subTest(file=name):
                (self.base / name).write_bytes(bytes([content[0] ^ 1]) + content[1:])
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    verify_base_model(self.base)
                (self.base / name).write_bytes(content)

    def test_missing_or_truncated_file_is_rejected(self):
        target = self.base / "model.safetensors"
        target.unlink()
        with self.assertRaisesRegex(ValueError, "Missing approved"):
            verify_base_model(self.base)
        target.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            verify_base_model(self.base)

    def test_extra_loader_files_cannot_override_verified_files(self):
        for name in ("added_tokens.json", "special_tokens_map.json", "adapter_config.json", "pytorch_model.bin"):
            with self.subTest(file=name):
                target = self.base / name
                target.write_text("{}")
                with self.assertRaisesRegex(ValueError, "Unapproved files"):
                    verify_base_model(self.base)
                target.unlink()

    def test_claimed_name_and_forged_cache_cannot_approve_wrong_weights(self):
        target = self.base / "model.safetensors"
        wrong = b"X" * len(self.contents[target.name])
        target.write_bytes(wrong)
        cache = self.base / ".cache/huggingface/download"
        cache.mkdir(parents=True)
        (cache / "model.safetensors.metadata").write_text(
            model_source.REVISION + "\n" + hashlib.sha256(wrong).hexdigest())
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_base_model(self.base)

    def test_invalid_source_fails_before_device_loader_and_output_creation(self):
        (self.base / "config.json").write_text("{}")
        args = SimpleNamespace(epochs=1, batch_size=2, accumulation_steps=8, max_length=64,
                               learning_rate=2e-5, warmup_ratio=0.1, smoke=False,
                               output_dir=self.root / "run", checkpoint=self.root / "checkpoint",
                               base_model=self.base, device="cuda")
        with patch("src.training.choose_device") as device, \
                patch("src.training.AutoModelForSequenceClassification.from_pretrained") as model, \
                patch("src.training.AutoTokenizer.from_pretrained") as tokenizer:
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                train(args)
            device.assert_not_called()
            model.assert_not_called()
            tokenizer.assert_not_called()
        self.assertFalse(args.output_dir.exists())
        self.assertFalse(args.checkpoint.exists())

    def test_source_record_is_saved_in_run_and_selected_checkpoint(self):
        # Exercise the real config/checkpoint writing path with one tiny CPU batch.
        # The fixture is deliberately not a BERT model or a formal experiment.
        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 3)

            def forward(self, input_ids):
                return SimpleNamespace(logits=self.linear(input_ids.float()))

        model = TinyClassifier()
        model.save_pretrained = Mock()
        tokenizer = Mock()
        rows = [{"id": str(i), "label": label} for i, label in enumerate(("negative", "neutral", "positive"))]
        batches = [{"input_ids": torch.tensor([[1.0], [2.0], [3.0]]), "labels": torch.tensor([0, 1, 2])}]
        split = self.root / "split.json"
        split.write_text('{"dataset_sha256": "test-fixture"}')
        args = SimpleNamespace(epochs=1, batch_size=3, accumulation_steps=1, max_length=64,
                               learning_rate=2e-5, warmup_ratio=0.1, smoke=False,
                               output_dir=self.root / "run", checkpoint=self.root / "checkpoint",
                               base_model=self.base, device="cpu", no_amp=True, seed=42,
                               data_dir=self.root, split=split, class_weighting="none",
                               no_gradient_checkpointing=True, preprocessing="whitespace-v1")
        with patch("src.training.AutoModelForSequenceClassification.from_pretrained", return_value=model), \
                patch("src.training.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                patch("src.training.load_split", return_value=(rows, rows, {"dataset_sha256": "test-fixture"})), \
                patch("src.training.loader", return_value=batches), patch("src.training.plot_history"), \
                patch("src.training.load_checkpoint", side_effect=lambda path, device: (
                    model.eval(), tokenizer, json.loads((path / "training_config.json").read_text()))), \
                redirect_stdout(io.StringIO()):
            train(args)
        run = json.loads((args.output_dir / "run_config.json").read_text())
        saved = json.loads((args.checkpoint / "training_config.json").read_text())
        self.assertEqual(run["base_model_source"], verify_base_model(self.base))
        self.assertEqual(saved["base_model_source"], run["base_model_source"])
        self.assertEqual(saved["training_code_sha256"], run["training_code_sha256"])
        self.assertEqual(run["training_code_sha256"]["src/model_source.py"],
                         hashlib.sha256(Path(model_source.__file__).read_bytes()).hexdigest())
        self.assertEqual(saved["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
