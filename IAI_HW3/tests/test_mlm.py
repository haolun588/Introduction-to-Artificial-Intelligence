import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoModelForSequenceClassification, BertConfig, BertForMaskedLM, BertForPreTraining, BertTokenizerFast

from src import mlm
from src.mlm import fixed_mask_batches, load_encoder_from_mlm, pretrain_mlm
from src.model_source import file_sha256
from src.training import train

LABELS = ("negative", "neutral", "positive")
WORDS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


def make_rows(ids):
    return [{"id": str(i), "label": LABELS[i % 3], "text": " ".join(WORDS[(i + k) % 10] for k in range(5 + i % 4))}
            for i in ids]


class MlmFixture(unittest.TestCase):
    """Tiny offline BERT, built once; verify_base_model / load_split are patched like the other tests."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.base = cls.root / "base"
        cls.base.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
        (cls.base / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
        BertTokenizerFast(vocab_file=str(cls.base / "vocab.txt")).save_pretrained(cls.base)
        torch.manual_seed(0)
        config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=32, max_position_embeddings=64)
        BertForPreTraining(config).save_pretrained(cls.base, safe_serialization=True)  # like the real checkpoint: has a pooler
        cls.source = {"directory": str(cls.base), "model": "tiny", "revision": "r"}
        cls.train_rows, cls.val_rows = make_rows(range(24)), make_rows(range(100, 106))
        cls.split = cls.root / "split.json"
        cls.split.write_text("{}")
        cls.calls = []
        cls.mlm_out, cls.mlm_ckpt = cls.root / "mlm_out", cls.root / "mlm_ckpt"
        cls.run_mlm(cls.mlm_out, cls.mlm_ckpt, epochs=2, record=cls.calls)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def mlm_args(cls, output, checkpoint, epochs=2, **extra):
        values = dict(split=cls.split, data_dir=cls.root, base_model=cls.base, output_dir=output,
                      checkpoint=checkpoint, epochs=epochs, learning_rate=5e-3, mask_prob=0.3,
                      warmup_ratio=0.06, weight_decay=0.01, batch_size=2, accumulation_steps=2, max_length=64,
                      seed=42, no_amp=True, no_gradient_checkpointing=True, device="cpu", smoke=False,
                      preprocessing="whitespace-v1")
        values.update(extra)
        return SimpleNamespace(**values)

    @classmethod
    def run_mlm(cls, output, checkpoint, epochs=2, record=None, **extra):
        real = mlm.collate_rows

        def spy(rows, *a, **k):
            if record is not None:
                record.append(([r["id"] for r in rows], k.get("shuffle", False)))
            return real(rows, *a, **k)

        with patch("src.mlm.verify_base_model", return_value=cls.source), \
                patch("src.mlm.load_split", return_value=(cls.train_rows, cls.val_rows, {"dataset_sha256": "f"})), \
                patch("src.mlm.collate_rows", side_effect=spy), redirect_stdout(io.StringIO()):
            pretrain_mlm(cls.mlm_args(output, checkpoint, epochs, **extra))

    def train_args(self, root, init_from, split=None):
        return SimpleNamespace(epochs=1, eval_interval=1.0, batch_size=2, accumulation_steps=2, max_length=64,
                               learning_rate=0.01, warmup_ratio=0.1, smoke=False, output_dir=root / "run",
                               checkpoint=root / "ckpt", base_model=self.base, device="cpu", no_amp=True,
                               seed=42, data_dir=root, split=self.split, class_weighting="sqrt-inverse",
                               no_gradient_checkpointing=True, preprocessing="whitespace-v1",
                               init_from=init_from)

    def run_train(self, init_from, source=None, manifest=None, allow_overlap=None):
        root = Path(tempfile.mkdtemp(dir=self.root))
        args = self.train_args(root, init_from)
        if allow_overlap is not None:
            args.allow_mlm_val_overlap = allow_overlap
        with patch("src.training.verify_base_model", return_value=source or self.source), \
                patch("src.training.load_split",
                      return_value=(self.train_rows, self.val_rows, manifest or {"dataset_sha256": "f"})), \
                patch("src.training.plot_history"), redirect_stdout(io.StringIO()):
            train(args)
        return (json.loads((args.output_dir / "run_config.json").read_text()),
                json.loads((args.checkpoint / "training_config.json").read_text()))

    def variant(self, edit):
        """Copy of the MLM checkpoint with mlm_config.json edited by `edit(dict)`."""
        target = Path(tempfile.mkdtemp(dir=self.root)) / "mlm"
        shutil.copytree(self.mlm_ckpt, target)
        path = target / "mlm_config.json"
        config = json.loads(path.read_text())
        edit(config)
        path.write_text(json.dumps(config))
        return target


class PretrainMlmTests(MlmFixture):
    def test_corpus_is_train_subset_only_and_validation_used_for_eval_only(self):
        train_ids = [r["id"] for r in self.train_rows]
        val_ids = [r["id"] for r in self.val_rows]
        self.assertEqual(self.calls[0], (train_ids, True))   # gradient corpus: training texts, shuffled
        self.assertEqual(self.calls[1], (val_ids, False))    # validation: fixed-mask eval loader only
        config = json.loads((self.mlm_ckpt / "mlm_config.json").read_text())
        self.assertEqual(config["mlm_text_ids"], train_ids)
        self.assertFalse(set(config["mlm_text_ids"]) & set(config["validation_ids"]))

    def test_eval_batch_size_defaults_and_applies_to_validation_loader_only(self):
        default = json.loads((self.mlm_ckpt / "mlm_config.json").read_text())
        self.assertEqual(default["eval_batch_size"], 2)
        self.assertIsNone(default["eval_batch_size_arg"])
        out, ckpt = Path(tempfile.mkdtemp(dir=self.root)) / "o", Path(tempfile.mkdtemp(dir=self.root)) / "c"
        sizes = []
        real = mlm.collate_rows

        def spy(rows, tokenizer, max_length, batch_size, *a, **k):
            sizes.append((k.get("shuffle", False), batch_size))
            return real(rows, tokenizer, max_length, batch_size, *a, **k)

        with patch("src.mlm.collate_rows", side_effect=spy):
            self.run_mlm(out, ckpt, epochs=1, eval_batch_size=3)
        self.assertEqual(sizes, [(True, 2), (False, 3)])
        config = json.loads((ckpt / "mlm_config.json").read_text())
        self.assertEqual((config["batch_size"], config["eval_batch_size"]), (2, 3))
        summary = json.loads((out / "summary.json").read_text())
        for key in ("total_train_seconds", "total_eval_seconds"):
            self.assertGreaterEqual(summary[key], 0)

    def test_records_and_artifacts(self):
        config = json.loads((self.mlm_ckpt / "mlm_config.json").read_text())
        for key in mlm.MLM_REQUIRED_CONFIG_KEYS + ("final_epoch", "final_validation_mlm_loss", "mask_prob"):
            self.assertIn(key, config)
        self.assertEqual(config["base_model_source"], self.source)
        self.assertEqual(config["split_sha256"], file_sha256(self.split))
        self.assertEqual((config["epochs"], config["learning_rate"], config["mask_prob"], config["warmup_ratio"],
                          config["weight_decay"]), (2, 5e-3, 0.3, 0.06, 0.01))
        self.assertEqual(set(config["training_code_sha256"]), set(mlm.MLM_CODE_FILES))
        self.assertEqual(json.loads((self.mlm_out / "run_config.json").read_text())["mlm_text_ids"],
                         config["mlm_text_ids"])
        history = json.loads((self.mlm_out / "history.json").read_text())
        self.assertEqual([r["epoch"] for r in history], [1, 2])
        for row in history:
            self.assertTrue({"train_mlm_loss", "validation_mlm_loss", "learning_rate", "seconds"} <= set(row))
        for name in ("curves.png", "environment.json", "summary.json"):
            self.assertTrue((self.mlm_out / name).is_file(), name)
        for name in ("model.safetensors", "mlm_summary.json", "tokenizer.json"):
            self.assertTrue((self.mlm_ckpt / name).is_file(), name)
        summary = json.loads((self.mlm_out / "summary.json").read_text())
        self.assertEqual(summary["model_sha256"], file_sha256(self.mlm_ckpt / "model.safetensors"))

    def test_validation_masks_are_fixed_and_rng_is_restored(self):
        tokenizer = BertTokenizerFast.from_pretrained(self.base)
        from transformers import DataCollatorForLanguageModeling
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.3)
        make = lambda: mlm.collate_rows(self.val_rows, tokenizer, 64, 2, collator, "whitespace-v1")
        state = torch.get_rng_state().clone()
        first, second = fixed_mask_batches(make(), 43), fixed_mask_batches(make(), 43)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a["input_ids"], b["input_ids"]) and torch.equal(a["labels"], b["labels"]))

    def test_zero_mask_final_window_is_skipped_not_crashed(self):
        # 25 texts, batch 2, accumulation 2 -> 13 batches, last window holds one single-text batch.
        # Force that batch to draw no mask: the window has no backward and must not reach scaler.unscale_/step.
        real_count = mlm.masked_count
        patcher = patch.object(type(self), "train_rows", make_rows(range(25)))
        patcher.start()
        self.addCleanup(patcher.stop)
        out, ckpt = self.root / "empty_out", self.root / "empty_ckpt"
        with patch("src.mlm.masked_count",
                   side_effect=lambda b: 0 if b["labels"].shape[0] == 1 else real_count(b)):
            self.run_mlm(out, ckpt, epochs=2)
        summary = json.loads((out / "summary.json").read_text())
        history = json.loads((out / "history.json").read_text())
        self.assertEqual(summary["empty_windows"], 2)          # one empty final window per epoch
        self.assertEqual([r["empty_windows"] for r in history], [1, 2])
        self.assertEqual(summary["optimizer_updates"], 2 * 6)  # the 6 non-empty windows per epoch still step
        self.assertEqual(summary["attempted_updates"], 2 * 6)
        self.assertEqual(summary["skipped_updates"], 0)

    def test_normal_run_has_no_empty_windows(self):
        summary = json.loads((self.mlm_out / "summary.json").read_text())
        self.assertEqual(summary["empty_windows"], 0)

    def test_refuses_existing_nonempty_paths(self):
        with self.assertRaisesRegex(ValueError, "already contains files"):
            self.run_mlm(self.mlm_out, self.root / "other_ckpt")
        with self.assertRaisesRegex(ValueError, "already contains files"):
            self.run_mlm(self.root / "other_out", self.mlm_ckpt)

    def test_encoder_is_copied_and_pooler_kept_from_base(self):
        torch.manual_seed(1)
        base_cls = AutoModelForSequenceClassification.from_pretrained(self.base, num_labels=3)
        torch.manual_seed(1)
        model = AutoModelForSequenceClassification.from_pretrained(self.base, num_labels=3)
        tokenizer = BertTokenizerFast.from_pretrained(self.base)
        load_encoder_from_mlm(model, self.mlm_ckpt, tokenizer)
        trained = BertForMaskedLM.from_pretrained(self.mlm_ckpt)
        self.assertTrue(torch.equal(model.bert.embeddings.word_embeddings.weight,
                                    trained.bert.embeddings.word_embeddings.weight))
        self.assertFalse(torch.equal(model.bert.embeddings.word_embeddings.weight,
                                     base_cls.bert.embeddings.word_embeddings.weight))
        self.assertTrue(torch.equal(model.bert.pooler.dense.weight, base_cls.bert.pooler.dense.weight))
        self.assertTrue(torch.equal(model.classifier.weight, base_cls.classifier.weight))


class AllDatasetCorpusTests(MlmFixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.all_calls = []
        cls.all_out, cls.all_ckpt = cls.root / "all_out", cls.root / "all_ckpt"
        cls.run_mlm(cls.all_out, cls.all_ckpt, epochs=2, record=cls.all_calls, mlm_corpus="all-dataset")

    def test_corpus_is_all_dataset_texts_without_validation_loader(self):
        all_ids = [r["id"] for r in self.train_rows + self.val_rows]
        self.assertEqual(self.all_calls, [(all_ids, True)])  # one shuffled gradient loader, no eval loader
        self.assertEqual(len(set(all_ids)), len(self.train_rows) + len(self.val_rows))

    def test_config_summary_and_history_records(self):
        all_ids = [r["id"] for r in self.train_rows + self.val_rows]
        config = json.loads((self.all_ckpt / "mlm_config.json").read_text())
        for key in mlm.MLM_REQUIRED_CONFIG_KEYS + ("final_epoch", "final_validation_mlm_loss", "mlm_corpus"):
            self.assertIn(key, config)
        self.assertEqual(config["mlm_corpus"], "all-dataset")
        self.assertEqual(config["mlm_text_ids"], all_ids)
        self.assertEqual(config["validation_ids"], [])
        self.assertEqual(config["validation_use"], "none; corpus is all dataset texts (no held-out text)")
        self.assertIsNone(config["final_validation_mlm_loss"])
        self.assertIsNone(json.loads((self.all_out / "summary.json").read_text())["final_validation_mlm_loss"])
        history = json.loads((self.all_out / "history.json").read_text())
        self.assertTrue(all(r["validation_mlm_loss"] is None for r in history))
        self.assertTrue((self.all_out / "curves.png").is_file())
        self.assertEqual(json.loads((self.mlm_ckpt / "mlm_config.json").read_text())["mlm_corpus"], "split-train")

    def test_smoke_subsets_merged_corpus(self):
        calls = []
        self.run_mlm(self.root / "sm_out", self.root / "sm_ckpt", epochs=2, record=calls, smoke=True,
                     mlm_corpus="all-dataset")
        ids = calls[0][0]
        self.assertEqual(len(calls), 1)
        self.assertLessEqual(len(ids), len(self.train_rows) + len(self.val_rows))
        self.assertTrue(set(ids) <= {r["id"] for r in self.train_rows + self.val_rows})

    def test_invalid_corpus_value_rejected(self):
        with self.assertRaisesRegex(ValueError, "mlm-corpus"):
            self.run_mlm(self.root / "bad_out", self.root / "bad_ckpt", mlm_corpus="nope")

    def test_plot_handles_all_none_validation(self):
        path = self.root / "plot.png"
        mlm.plot_mlm_history([{"epoch": 1, "train_mlm_loss": 2.0, "validation_mlm_loss": None},
                              {"epoch": 2, "train_mlm_loss": 1.5, "validation_mlm_loss": None}], path)
        self.assertTrue(path.is_file())

    def test_finetune_from_all_dataset_requires_flag_and_records_non_strict(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.run_train(self.all_ckpt)
        run, saved = self.run_train(self.all_ckpt, allow_overlap=True)
        n = len(self.val_rows)
        self.assertEqual(run["init_from"]["validation_overlap"], {"allowed": True, "n_overlap_ids": n, "strict_oof": False})
        self.assertEqual(run["init_from"]["mlm_run_config"]["mlm_corpus"], "all-dataset")
        self.assertEqual(run["init_from"]["mlm_run_config"]["n_texts"], len(self.train_rows) + n)
        self.assertEqual(saved["init_from"], run["init_from"])


class InitFromTests(MlmFixture):
    def assert_aborts_before_dirs(self, init_from, pattern, **kwargs):
        before = set(self.root.iterdir())
        with self.assertRaisesRegex(ValueError, pattern):
            self.run_train(init_from, **kwargs)
        created = [p for p in set(self.root.iterdir()) - before if (p / "run").exists() or (p / "ckpt").exists()]
        self.assertEqual(created, [])

    def test_without_init_from_record_is_null(self):
        run, saved = self.run_train(None)
        self.assertIsNone(run["init_from"])
        self.assertIsNone(saved["init_from"])
        self.assertNotIn("src/mlm.py", run["training_code_sha256"])

    def test_with_init_from_records_provenance(self):
        run, saved = self.run_train(self.mlm_ckpt)
        record = run["init_from"]
        self.assertEqual(saved["init_from"], record)
        self.assertEqual(record["path"], str(self.mlm_ckpt))
        self.assertEqual(record["model_sha256"], file_sha256(self.mlm_ckpt / "model.safetensors"))
        self.assertEqual(record["mlm_run_config"],
                         {"epochs": 2, "learning_rate": 5e-3, "mask_prob": 0.3,
                          "split_sha256": file_sha256(self.split), "n_texts": len(self.train_rows),
                          "mlm_corpus": "split-train"})
        self.assertIn("src/mlm.py", run["training_code_sha256"])
        self.assertEqual(run["train_ids"], [r["id"] for r in self.train_rows])

    def test_validation_overlap_aborts(self):
        bad = self.variant(lambda c: c["mlm_text_ids"].append(self.val_rows[0]["id"]))
        self.assert_aborts_before_dirs(bad, "overlaps")

    def test_not_subset_of_train_aborts(self):
        bad = self.variant(lambda c: c["mlm_text_ids"].append("not-a-train-id"))
        self.assert_aborts_before_dirs(bad, "not a subset")

    def test_base_model_mismatch_aborts(self):
        self.assert_aborts_before_dirs(self.mlm_ckpt, "different base model", source={**self.source, "revision": "x"})

    def test_dataset_hash_mismatch_aborts(self):
        self.assert_aborts_before_dirs(self.mlm_ckpt, "dataset hash differs", manifest={"dataset_sha256": "other"})

    def test_smoke_mlm_checkpoint_rejected_for_non_smoke_train(self):
        bad = self.variant(lambda c: c.update(smoke=True))
        self.assert_aborts_before_dirs(bad, "smoke MLM checkpoint")

    def test_missing_smoke_field_aborts(self):
        bad = self.variant(lambda c: c.pop("smoke"))
        self.assert_aborts_before_dirs(bad, "lacks fields")

    def test_smoke_mlm_checkpoint_accepted_for_smoke_train(self):
        ok = self.variant(lambda c: c.update(smoke=True))
        info = mlm.check_init_from(ok, self.source, {"dataset_sha256": "f"}, self.train_rows, self.val_rows, True)
        self.assertEqual(info["path"], str(ok))

    def test_record_has_validation_overlap_and_corpus_for_strict_run(self):
        run, saved = self.run_train(self.mlm_ckpt)
        self.assertEqual(run["init_from"]["validation_overlap"],
                         {"allowed": False, "n_overlap_ids": 0, "strict_oof": True})
        self.assertEqual(run["init_from"]["mlm_run_config"]["mlm_corpus"], "split-train")
        self.assertEqual(saved["init_from"], run["init_from"])

    def test_overlap_still_aborts_when_flag_false(self):
        bad = self.variant(lambda c: c["mlm_text_ids"].append(self.val_rows[0]["id"]))
        self.assert_aborts_before_dirs(bad, "overlaps", allow_overlap=False)

    def test_overlap_allowed_with_flag_and_recorded(self):
        n_val = 3
        ok = self.variant(lambda c: c["mlm_text_ids"].extend(r["id"] for r in self.val_rows[:n_val]))
        run, saved = self.run_train(ok, allow_overlap=True)
        self.assertEqual(run["init_from"]["validation_overlap"],
                         {"allowed": True, "n_overlap_ids": n_val, "strict_oof": False})
        self.assertEqual(saved["init_from"], run["init_from"])

    def test_flag_without_actual_overlap_is_still_strict(self):
        info = mlm.check_init_from(self.mlm_ckpt, self.source, {"dataset_sha256": "f"}, self.train_rows,
                                   self.val_rows, False, allow_val_overlap=True)
        self.assertEqual(info["validation_overlap"], {"allowed": True, "n_overlap_ids": 0, "strict_oof": True})

    def test_flag_still_rejects_ids_outside_dataset(self):
        bad = self.variant(lambda c: c["mlm_text_ids"].append("not-in-dataset"))
        self.assert_aborts_before_dirs(bad, "outside", allow_overlap=True)

    def test_flag_still_checks_other_guards(self):
        bad = self.variant(lambda c: c["mlm_text_ids"].append(c["mlm_text_ids"][0]))
        self.assert_aborts_before_dirs(bad, "duplicates", allow_overlap=True)

    def test_legacy_config_without_mlm_corpus_is_split_train(self):
        legacy = self.variant(lambda c: c.pop("mlm_corpus"))
        info = mlm.check_init_from(legacy, self.source, {"dataset_sha256": "f"}, self.train_rows, self.val_rows)
        self.assertEqual(info["mlm_run_config"]["mlm_corpus"], "split-train")
        self.assertNotIn("mlm_corpus", mlm.MLM_REQUIRED_CONFIG_KEYS)

    def test_missing_summary_aborts(self):
        bad = self.variant(lambda c: None)
        (bad / "mlm_summary.json").unlink()
        self.assert_aborts_before_dirs(bad, "did not finish")


if __name__ == "__main__":
    unittest.main()
