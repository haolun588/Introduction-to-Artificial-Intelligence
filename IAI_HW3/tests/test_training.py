import csv
import hashlib
import json
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from src.training import (accumulation_sample_count, class_weights, evaluate, evaluation_points,
                          load_split, metrics, train, training_loss)


class TrainingTests(unittest.TestCase):
    def test_quarter_weights_are_milder_and_sample_normalized(self):
        counts = torch.tensor([16., 4., 1.])
        rows = [{"label": label} for label, count in zip(("negative", "neutral", "positive"), (16, 4, 1)) for _ in range(count)]
        weights = class_weights(rows, "quarter-inverse")
        torch.testing.assert_close(weights / weights[0], torch.tensor([1., 2**0.5, 2.]))
        self.assertAlmostEqual(float((weights * counts).sum() / counts.sum()), 1., places=6)
        strong = class_weights(rows, "sqrt-inverse")
        self.assertLess(float(weights[-1] / weights[0]), float(strong[-1] / strong[0]))
        with self.assertRaises(ValueError):
            class_weights([{"label": "negative"}], "quarter-inverse")

    def test_half_epoch_points_respect_accumulation_and_short_epochs(self):
        self.assertEqual(evaluation_points(4050, 8, 0.5), {2032: 0.5, 4050: 1.0})
        self.assertEqual(evaluation_points(7, 2, 0.5), {4: 0.5, 7: 1.0})
        self.assertEqual(evaluation_points(5, 8, 0.5), {5: 1.0})
        self.assertEqual(evaluation_points(4050, 8, 1.0), {4050: 1.0})
        with self.assertRaises(ValueError):
            evaluation_points(0, 8, 0.5)

    def test_evaluation_preserves_rng_and_mode_even_on_failure(self):
        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fail = False

            def forward(self, input_ids):
                self.assert_eval = not self.training
                torch.rand(5)  # Deliberately consume RNG to test isolation.
                if self.fail:
                    raise RuntimeError("fixture failure")
                return SimpleNamespace(logits=input_ids)

        model = Classifier().train()
        batches = [{"input_ids": torch.eye(3), "labels": torch.arange(3)}]
        rng = torch.get_rng_state().clone()
        self.assertEqual(evaluate(model, batches, torch.device("cpu"))["macro_f1"], 1.0)
        self.assertTrue(model.assert_eval and model.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        model.fail = True
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            evaluate(model, batches, torch.device("cpu"))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        model.fail = False
        model.eval()
        evaluate(model, batches, torch.device("cpu"))
        self.assertFalse(model.training)

    def test_class_weights_use_training_counts_and_mean_one(self):
        rows = [{'label': label} for label, count in [('negative', 9), ('neutral', 4), ('positive', 1)] for _ in range(count)]
        weights = class_weights(rows, 'sqrt-inverse')
        torch.testing.assert_close(weights / weights[0], torch.tensor([1., 1.5, 3.]))
        self.assertAlmostEqual(float((weights * torch.tensor([9, 4, 1])).sum() / 14), 1.0, places=6)
        self.assertIsNone(class_weights(rows, 'none'))
        with self.assertRaises(ValueError):
            class_weights([{'label': 'negative'}], 'sqrt-inverse')

    def test_weighted_gradients_match_full_windows_including_partial_batch(self):
        torch.manual_seed(42)
        targets = torch.tensor([0, 0, 1, 2, 0, 1, 2, 2, 0, 1, 2])
        weights = torch.tensor([0.7, 1.5, 1.6])
        expected = torch.randn(11, 3, requires_grad=True)
        actual = expected.detach().clone().requires_grad_(True)
        for start in (0, 6):
            end = min(start+6, 11)
            log_probs = expected[start:end].log_softmax(-1)
            loss = -(log_probs[torch.arange(end-start), targets[start:end]] * weights[targets[start:end]]).sum() / (end-start)
            loss.backward()
        for index, start in enumerate(range(0, 11, 2)):
            training_loss(actual[start:start+2], targets[start:start+2], weights,
                          accumulation_sample_count(index, 2, 3, 11)).backward()
        torch.testing.assert_close(actual.grad, expected.grad)
        plain = training_loss(actual, targets, None, len(targets))
        torch.testing.assert_close(plain, torch.nn.functional.cross_entropy(actual, targets))

    def test_label_smoothing_zero_matches_legacy_loss_and_gradient(self):
        torch.manual_seed(3)
        targets = torch.tensor([0, 1, 2, 2, 0])
        weights = torch.tensor([0.7, 1.5, 1.6])
        for w in (weights, None):
            a = torch.randn(5, 3, requires_grad=True)
            b = a.detach().clone().requires_grad_(True)
            legacy = torch.nn.functional.cross_entropy(a, targets, weight=w, reduction="sum") / 4
            new = training_loss(b, targets, w, 4, 0.0)
            self.assertTrue(torch.equal(legacy, new))
            legacy.backward()
            new.backward()
            self.assertTrue(torch.equal(a.grad, b.grad))
            self.assertTrue(torch.equal(new, training_loss(b, targets, w, 4)))

    def test_label_smoothing_matches_formula_with_true_class_weights(self):
        torch.manual_seed(5)
        eps = 0.1
        targets = torch.tensor([0, 1, 2, 2, 0])
        weights = torch.tensor([0.7, 1.5, 1.6])
        logits = torch.randn(5, 3)
        nll = -logits.log_softmax(-1)
        per = (1 - eps) * nll[torch.arange(5), targets] + eps * nll.mean(-1)
        expected = (weights[targets] * per).sum() / 4
        torch.testing.assert_close(training_loss(logits, targets, weights, 4, eps), expected)
        torch.testing.assert_close(training_loss(logits, targets, None, 4, eps), per.sum() / 4)
        # PyTorch's weight= plus label_smoothing= weights the smoothing term by target class; ours must not.
        pytorch_combo = torch.nn.functional.cross_entropy(
            logits, targets, weight=weights, reduction="sum", label_smoothing=eps) / 4
        self.assertFalse(torch.allclose(training_loss(logits, targets, weights, 4, eps), pytorch_combo))

    def test_label_smoothing_out_of_range_is_rejected(self):
        for value in (-0.1, 1.0, 1.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                train(SimpleNamespace(label_smoothing=value))

    def test_evaluation_loss_ignores_label_smoothing(self):
        class Model(torch.nn.Module):
            def forward(self, input_ids):
                return SimpleNamespace(logits=torch.tensor([[2., 0., 0.], [0., 1., 0.]]))

        batches = [{"input_ids": torch.zeros(2, 1, dtype=torch.long), "labels": torch.tensor([0, 2])}]
        expected = torch.nn.functional.cross_entropy(
            torch.tensor([[2., 0., 0.], [0., 1., 0.]]), torch.tensor([0, 2])).item()
        self.assertAlmostEqual(evaluate(Model(), batches, torch.device("cpu"))["loss"], expected, places=6)

    def test_accumulation_matches_large_batch_with_partial_window(self):
        torch.manual_seed(7)
        inputs, targets = torch.randn(11, 4), torch.randn(11, 2)
        reference = torch.nn.Linear(4, 2)
        accumulated = torch.nn.Linear(4, 2)
        accumulated.load_state_dict(reference.state_dict())
        optimizers = [torch.optim.SGD(m.parameters(), lr=0.1) for m in (reference, accumulated)]
        # Two windows: 6 samples, then 5 (including a final micro-batch of 1).
        for start in (0, 6):
            optimizers[0].zero_grad()
            stop = min(start + 6, len(inputs))
            loss = (reference(inputs[start:stop]) - targets[start:stop]).square().sum() / (stop-start)
            loss.backward()
            optimizers[0].step()
        optimizers[1].zero_grad()
        for index, start in enumerate(range(0, len(inputs), 2)):
            loss = (accumulated(inputs[start:start+2]) - targets[start:start+2]).square().sum()
            (loss / accumulation_sample_count(index, 2, 3, len(inputs))).backward()
            if (index+1) % 3 == 0 or start+2 >= len(inputs):
                optimizers[1].step()
                optimizers[1].zero_grad()
        for expected, actual in zip(reference.parameters(), accumulated.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_macro_f1_includes_missing_classes(self):
        result = metrics([0, 0], [0, 0])
        self.assertAlmostEqual(result['macro_f1'], 1/3)
        self.assertEqual(len(result['confusion_matrix']), 3)

    def test_split_rejects_overlap_and_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            source, split = directory / 'dataset.csv', directory / 'split.json'
            with source.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'text', 'label'])
                writer.writeheader()
                writer.writerow({'id': '1', 'text': 'example', 'label': 'negative'})
            manifest = {'dataset_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                        'train_ids': ['1'], 'validation_ids': ['1']}
            split.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'partition'):
                load_split(directory, split)
            source.write_text(source.read_text() + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_split(directory, split)


class TrainingLoopIntegrationTests(unittest.TestCase):
    def run_fixture(self, root, interval, skip_first=False, eval_batch_size=None, loader_calls=None):
        """Real tiny dropout training; synthetic tied selection scores; no BERT."""
        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.4)
                self.linear = torch.nn.Linear(2, 3)

            def forward(self, input_ids):
                return SimpleNamespace(logits=self.linear(self.dropout(input_ids)))

            def save_pretrained(self, path, **kwargs):
                torch.save(self.state_dict(), path / "fixture.pt")

        models = []
        def create_model(*args, **kwargs):
            model = Classifier()
            models.append(model)
            return model

        tokenizer = Mock()
        def reload_model(path, device):
            model = Classifier()
            model.load_state_dict(torch.load(path / "fixture.pt", weights_only=True))
            return model.eval(), tokenizer, json.loads((path / "training_config.json").read_text())

        labels = ("negative", "neutral", "positive")
        train_rows = [{"id": str(i), "label": labels[i % 3]} for i in range(11)]
        val_rows = [{"id": str(i + 100), "label": labels[i % 3]} for i in range(6)]
        def batches(rows, tokenizer, max_length, batch_size, shuffle=False, seed=42, **kwargs):
            if loader_calls is not None:
                loader_calls.append((rows is val_rows, shuffle, batch_size))
            data = [{"input_ids": torch.tensor([i / 10, 1.0]), "labels": labels.index(r["label"])}
                    for i, r in enumerate(rows)]
            result = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                                  generator=torch.Generator().manual_seed(seed))
            result.is_validation = rows is val_rows
            return result

        def tied_evaluate(model, batches, device, amp=False):
            result = evaluate(model, batches, device, amp)
            if batches.is_validation:
                # Force ties to test retaining the first half-epoch checkpoint.
                result["macro_f1"] = 0.5
            return result

        class SkipFirstScaler:
            def __init__(self, *args, **kwargs):
                self.attempts, self.value = 0, 2.0
            def scale(self, loss): return loss
            def unscale_(self, optimizer): pass
            def get_scale(self): return self.value
            def step(self, optimizer):
                self.attempts += 1
                if self.attempts > 1: optimizer.step()
            def update(self):
                if self.attempts == 1: self.value = 1.0

        root.mkdir()
        split = root / "split.json"
        split.write_text("{}")
        args = SimpleNamespace(epochs=2, eval_interval=interval, batch_size=2, accumulation_steps=3,
                               max_length=64, learning_rate=0.03, warmup_ratio=0.25, smoke=False,
                               output_dir=root / "run", checkpoint=root / "checkpoint", base_model=root / "base",
                               device="cpu", no_amp=True, seed=42, data_dir=root, split=split,
                               class_weighting="sqrt-inverse", no_gradient_checkpointing=True, preprocessing="whitespace-v1")
        if eval_batch_size is not None:
            args.eval_batch_size = eval_batch_size
        original_scaler = torch.amp.GradScaler
        with patch("src.training.verify_base_model", return_value={"directory": str(args.base_model), "model": "fixture", "revision": "fixture"}), \
                patch("src.training.AutoModelForSequenceClassification.from_pretrained", side_effect=create_model), \
                patch("src.training.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                patch("src.training.load_split", return_value=(train_rows, val_rows, {"dataset_sha256": "fixture"})), \
                patch("src.training.loader", side_effect=batches), patch("src.training.plot_history"), \
                patch("src.training.load_checkpoint", side_effect=reload_model), \
                patch("src.training.evaluate", side_effect=tied_evaluate), \
                patch("src.training.torch.amp.GradScaler", SkipFirstScaler if skip_first else original_scaler), \
                redirect_stdout(io.StringIO()):
            train(args)
        return (models[0].state_dict(),
                json.loads((args.output_dir / "history.json").read_text()),
                json.loads((args.output_dir / "summary.json").read_text()))

    def test_mid_epoch_evaluation_does_not_change_training_and_retains_first_best(self):
        with tempfile.TemporaryDirectory() as folder:
            full, _, full_summary = self.run_fixture(Path(folder) / "full", 1.0)
            half, history, summary = self.run_fixture(Path(folder) / "half", 0.5)
        for name in full:
            self.assertTrue(torch.equal(full[name], half[name]), name)
        self.assertEqual([r["epoch"] for r in history], [0.5, 1.0, 1.5, 2.0])
        self.assertEqual([r["optimizer_updates"] for r in history], [1, 2, 3, 4])
        self.assertEqual(summary["best_epoch"], 0.5)
        self.assertEqual(full_summary["best_epoch"], 1.0)
        self.assertTrue(summary["reload_verified"])
        self.assertEqual(summary["scheduler_steps"], 4)
        self.assertEqual(summary["skipped_updates"], 0)
        self.assertAlmostEqual(history[1]["next_learning_rate"], 0.02)
        self.assertEqual(history[-1]["next_learning_rate"], 0)

    def test_eval_batch_size_defaults_to_64_for_evaluation_loaders_only(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            _, history, summary = self.run_fixture(Path(folder) / "default", 1.0, loader_calls=calls)
            config = json.loads((Path(folder) / "default" / "run" / "run_config.json").read_text())
        self.assertEqual([size for _, shuffle, size in calls if shuffle], [2])    # training loader keeps batch_size
        self.assertEqual([size for _, shuffle, size in calls if not shuffle], [64, 64])  # train_eval and validation
        self.assertEqual(config["eval_batch_size"], 64)
        self.assertIsNone(config["eval_batch_size_arg"])
        for key in ("total_train_seconds", "total_eval_seconds"):
            self.assertGreaterEqual(summary[key], 0)
        self.assertTrue(all(r["train_seconds"] >= 0 and r["eval_seconds"] >= 0 for r in history))

    def test_eval_batch_size_applies_only_to_evaluation_loaders(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            full, _, _ = self.run_fixture(Path(folder) / "base", 1.0)
            fast, _, _ = self.run_fixture(Path(folder) / "fast", 1.0, eval_batch_size=5, loader_calls=calls)
            config = json.loads((Path(folder) / "fast" / "run" / "run_config.json").read_text())
        self.assertEqual([size for _, shuffle, size in calls if shuffle], [2])
        self.assertEqual([size for _, shuffle, size in calls if not shuffle], [5, 5])
        self.assertEqual(config["eval_batch_size"], 5)
        self.assertEqual(config["batch_size"], 2)
        for name in full:  # evaluation batch size must not change training
            self.assertTrue(torch.equal(full[name], fast[name]), name)

    def test_overflow_skip_does_not_advance_scheduler(self):
        with tempfile.TemporaryDirectory() as folder:
            _, history, summary = self.run_fixture(Path(folder) / "skip", 0.5, skip_first=True)
        self.assertEqual(summary["attempted_updates"], 4)
        self.assertEqual(summary["optimizer_updates"], 3)
        self.assertEqual(summary["scheduler_steps"], 3)
        self.assertEqual(summary["skipped_updates"], 1)
        self.assertAlmostEqual(history[-1]["next_learning_rate"], 0.01)


if __name__ == '__main__':
    unittest.main()
