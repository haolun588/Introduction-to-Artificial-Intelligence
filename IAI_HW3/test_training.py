import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from training import accumulation_sample_count, class_weights, load_split, metrics, training_loss


class TrainingTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
