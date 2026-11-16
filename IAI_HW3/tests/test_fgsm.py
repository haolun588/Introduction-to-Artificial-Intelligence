"""Compare FGSM parameter gradients against an independent joint-loss graph."""
import copy
import unittest

import torch
from transformers import BertConfig, BertForSequenceClassification

from src.training import fgsm_backward, fgsm_delta, training_loss


class FGSMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = BertForSequenceClassification(BertConfig(
            vocab_size=20, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=24, num_labels=3, hidden_dropout_prob=0,
            attention_probs_dropout_prob=0)).train()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.batch = {"input_ids": torch.tensor([[1, 4, 5, 2, 0], [1, 6, 2, 0, 0], [1, 7, 8, 9, 2]]),
                      "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])}
        self.labels = torch.tensor([0, 1, 2])
        self.weights = torch.tensor([0.7, 1.4, 1.1])

    def test_sign_mask_bounds_and_nonfinite_rejection(self):
        grad = torch.arange(-7., 8.).reshape(1, 5, 3).requires_grad_()
        # Include an ordinary token with attention=0 to test padding independently.
        delta = fgsm_delta(grad, torch.tensor([[1, 4, 5, 6, 2]]),
                           torch.tensor([[1, 1, 1, 0, 1]]), [0, 1, 2], 0.003)
        self.assertFalse(delta.requires_grad)
        self.assertEqual(torch.count_nonzero(delta[:, [0, 3, 4]]), 0)
        torch.testing.assert_close(delta[:, 1:3], grad[:, 1:3].sign() * 0.003)
        for epsilon in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fgsm_delta(grad, self.batch["input_ids"][:1], self.batch["attention_mask"][:1], [], epsilon)
        with self.assertRaises(ValueError):
            fgsm_delta(grad * float("nan"), self.batch["input_ids"][:1], self.batch["attention_mask"][:1], [], .01)

    def test_accumulated_gradients_match_independent_joint_loss(self):
        reference = copy.deepcopy(self.model)
        initial = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        embed = reference.get_input_embeddings()(self.batch["input_ids"])
        clean = training_loss(reference(inputs_embeds=embed, attention_mask=self.batch["attention_mask"]).logits,
                              self.labels, self.weights, 3)
        grad, = torch.autograd.grad(clean, embed, retain_graph=True)
        mask = (self.batch["input_ids"] > 2).unsqueeze(-1)
        delta = .003 * grad.detach().sign() * mask
        adv = training_loss(reference(inputs_embeds=embed + delta, attention_mask=self.batch["attention_mask"]).logits,
                            self.labels, self.weights, 3)
        ((clean + adv) / 2).backward()
        # A partial last microbatch must still divide by the full window's 3 samples.
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        for start, stop in ((0, 2), (2, 3)):
            fgsm_backward(self.model, {k: v[start:stop] for k, v in self.batch.items()},
                          self.labels[start:stop], self.weights, 3, scaler, .003, [0, 1, 2])
        for (name, actual), expected in zip(self.model.named_parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-7, rtol=2e-4, msg=name)
            torch.testing.assert_close(actual, initial[name], atol=0, rtol=0)

    def test_zero_attack_matches_clean_and_preserves_clean_inference(self):
        reference = copy.deepcopy(self.model)
        clean = training_loss(reference(**self.batch).logits, self.labels, self.weights, 3)
        clean.backward()
        fgsm_backward(self.model, self.batch, self.labels, self.weights, 3,
                      torch.amp.GradScaler("cuda", enabled=False), 0, [0, 1, 2])
        for actual, expected in zip(self.model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-7, rtol=2e-4)
        with torch.no_grad():
            torch.testing.assert_close(self.model.eval()(**self.batch).logits,
                                       reference.eval()(**self.batch).logits, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for FP16 scaler check")
    def test_amp_scaling_and_accumulation_match_fp32_gradients(self):
        reference = copy.deepcopy(self.model).cuda()
        actual = self.model.cuda()
        batch = {k: v.cuda() for k, v in self.batch.items()}
        labels, weights = self.labels.cuda(), self.weights.cuda()
        fgsm_backward(reference, batch, labels, weights, 3,
                      torch.amp.GradScaler("cuda", enabled=False), .003, [0, 1, 2])
        scaler = torch.amp.GradScaler("cuda")
        optimizer = torch.optim.SGD(actual.parameters(), lr=.1)
        for start, stop in ((0, 2), (2, 3)):
            fgsm_backward(actual, {k: v[start:stop] for k, v in batch.items()},
                          labels[start:stop], weights, 3, scaler, .003, [0, 1, 2], amp=True)
        scaler.unscale_(optimizer)
        a = torch.cat([p.grad.flatten() for p in actual.parameters()])
        b = torch.cat([p.grad.flatten() for p in reference.parameters()])
        self.assertTrue(torch.isfinite(a).all())
        self.assertLess(((a - b).norm() / b.norm()).item(), .02)
        self.assertGreater(actual.get_input_embeddings().weight.grad.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
