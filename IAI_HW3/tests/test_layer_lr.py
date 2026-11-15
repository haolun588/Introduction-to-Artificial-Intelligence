"""Check real BERT parameter coverage and optimizer/scheduler behavior on CPU."""
import copy
import unittest

import torch
from transformers import BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup

from src.training import optimizer_groups


class LayerLearningRateTests(unittest.TestCase):
    def model(self):
        return BertForSequenceClassification(BertConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=12,
            num_attention_heads=2, num_labels=3, max_position_embeddings=16))

    def test_all_parameters_once_and_exact_depth_rates(self):
        model = self.model()
        groups = optimizer_groups(model, 2e-5, 0.9)
        params = [p for g in groups for p in g["params"]]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters()})
        rates = {n: g["lr"] for g in groups for n in g["parameter_names"]}
        self.assertEqual(rates["classifier.bias"], 2e-5)
        self.assertEqual(rates["bert.pooler.dense.weight"], 2e-5)
        self.assertEqual(rates["bert.encoder.layer.11.output.LayerNorm.bias"], 2e-5)
        self.assertAlmostEqual(rates["bert.encoder.layer.0.attention.self.query.weight"], 2e-5 * .9**11)
        self.assertAlmostEqual(rates["bert.embeddings.word_embeddings.weight"], 2e-5 * .9**12)
        optimizer = torch.optim.AdamW(groups, weight_decay=.01, foreach=False)
        initial = [g["lr"] for g in optimizer.param_groups]
        scheduler = get_linear_schedule_with_warmup(optimizer, 2, 6)
        for factor in (0.5, 1., .75, .5, .25, 0.):
            optimizer.step()
            scheduler.step()
            for group, rate in zip(optimizer.param_groups, initial):
                self.assertAlmostEqual(group["lr"], rate * factor)
                self.assertEqual(group["weight_decay"], .01)

    def test_uniform_path_matches_original_adamw_updates(self):
        model = self.model()
        other = copy.deepcopy(model)
        old = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01, foreach=False)
        new = torch.optim.AdamW(optimizer_groups(other, 2e-5), lr=2e-5, weight_decay=.01, foreach=False)
        for _ in range(3):
            for p, q in zip(model.parameters(), other.parameters()):
                p.grad = torch.ones_like(p)
                q.grad = p.grad.clone()
            old.step()
            new.step()
        for p, q in zip(model.parameters(), other.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_invalid_decay_and_unknown_parameters_fail_closed(self):
        model = self.model()
        for decay in (0., -1., 1.1, float("nan")):
            with self.assertRaises(ValueError):
                optimizer_groups(model, 2e-5, decay)
        model.extra = torch.nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(ValueError, "Unassigned"):
            optimizer_groups(model, 2e-5, .9)


if __name__ == "__main__":
    unittest.main()
