import unittest
from types import SimpleNamespace

from main import build_parser
from src.training import gradient_checkpointing_enabled, resolve_eval_batch_size


def parse(*argv):
    return build_parser().parse_args(list(argv))


class CliDefaultsTests(unittest.TestCase):
    def test_train_eval_batch_size_defaults_to_64_and_explicit_value_wins(self):
        args = parse("train")
        self.assertIsNone(args.eval_batch_size)
        self.assertEqual(resolve_eval_batch_size(args), 64)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(resolve_eval_batch_size(parse("train", "--eval-batch-size", "8")), 8)

    def test_pretrain_mlm_eval_batch_size_arg_stays_none_so_it_follows_batch_size(self):
        # src.mlm resolves None to --batch-size itself (covered by test_mlm); the parser must not inject 64.
        args = parse("pretrain-mlm", "--batch-size", "4")
        self.assertIsNone(args.eval_batch_size)

    def test_gradient_checkpointing_off_by_default_and_flags_work(self):
        self.assertFalse(gradient_checkpointing_enabled(parse("train")))
        self.assertFalse(gradient_checkpointing_enabled(parse("pretrain-mlm")))
        self.assertTrue(gradient_checkpointing_enabled(parse("train", "--gradient-checkpointing")))
        self.assertTrue(gradient_checkpointing_enabled(parse("pretrain-mlm", "--gradient-checkpointing")))
        self.assertFalse(gradient_checkpointing_enabled(parse("train", "--no-gradient-checkpointing")))

    def test_legacy_namespace_attribute_still_honoured(self):
        self.assertFalse(gradient_checkpointing_enabled(SimpleNamespace(no_gradient_checkpointing=True)))
        self.assertTrue(gradient_checkpointing_enabled(SimpleNamespace(no_gradient_checkpointing=False)))
        self.assertTrue(gradient_checkpointing_enabled(SimpleNamespace(gradient_checkpointing=True, no_gradient_checkpointing=True)))


if __name__ == "__main__":
    unittest.main()
