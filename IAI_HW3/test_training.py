import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from training import accumulation_sample_count, load_split, metrics


class TrainingTests(unittest.TestCase):
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
