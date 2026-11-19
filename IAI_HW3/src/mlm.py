"""Per-fold masked-LM continued pre-training on the training subset only, plus init-from guards for `train`.

The corpus is the fold's training texts (same preprocessing/tokenizer/max_length as fine-tuning). Validation
texts are used only to compute a fixed-mask MLM eval loss per epoch (no gradient). No labels, no benchmark.
With `mlm_corpus="all-dataset"` the corpus is every dataset text (train + validation of the given split): there is no
held-out text, so no MLM eval is done and the downstream OOF is NOT strict (see `check_init_from(allow_val_overlap)`).
"""

import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          get_linear_schedule_with_warmup)

from .model_source import file_sha256, verify_base_model
from .training import (ROOT, TextDataset, choose_device, load_split, smoke_subset, sync_cuda, write_json)

MLM_CODE_FILES = ("main.py", "src/__init__.py", "src/training.py", "src/eda.py", "src/preprocessing.py",
                  "src/model_source.py", "src/mlm.py")
MLM_REQUIRED_FILES = ("model.safetensors", "config.json", "mlm_config.json", "mlm_summary.json")
MLM_REQUIRED_CONFIG_KEYS = ("base_model_source", "split_path", "split_sha256", "dataset_sha256", "mlm_text_ids",
                            "validation_ids", "epochs", "learning_rate", "mask_prob", "warmup_ratio",
                            "weight_decay", "batch_size", "accumulation_steps", "max_length", "seed",
                            "preprocessing", "training_code_sha256", "environment", "smoke")


def collate_rows(rows, tokenizer, max_length, batch_size, collator, preprocessing, shuffle=False, seed=42):
    return DataLoader(TextDataset(rows, tokenizer, max_length, labeled=False, preprocessing=preprocessing),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed), collate_fn=collator)


def masked_count(batch):
    return int((batch["labels"] != -100).sum())


def fixed_mask_batches(loader_, seed):
    """Materialize validation batches once under a fixed RNG so every epoch sees identical masks."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return list(loader_)


@torch.inference_mode()
def evaluate_mlm(model, batches, device, amp=False):
    """Mean MLM loss per masked token over fixed batches."""
    was_training = model.training
    try:
        model.eval()
        total, count = 0.0, 0
        for batch in batches:
            n = masked_count(batch)
            if n == 0:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = model(**batch).loss
            total += loss.float().item() * n
            count += n
        if count == 0:
            raise ValueError("No masked tokens in the evaluation batches")
        return total / count
    finally:
        model.train(was_training)


def plot_mlm_history(history, path):
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    epochs = [r["epoch"] for r in history]
    ax.plot(epochs, [r["train_mlm_loss"] for r in history], "o-", label="train")
    if any(r["validation_mlm_loss"] is not None for r in history):
        ax.plot(epochs, [r["validation_mlm_loss"] for r in history], "o-", label="validation")
    ax.set(xlabel="Epoch", ylabel="MLM loss", xticks=epochs)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pretrain_mlm(args):
    if args.split is None or args.output_dir is None or args.checkpoint is None:
        raise ValueError("pretrain-mlm requires --split, --output-dir and --checkpoint")
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.max_length) < 1:
        raise ValueError("Epochs, batch size, accumulation and max length must be positive")
    eval_batch_size = args.batch_size if getattr(args, "eval_batch_size", None) is None else args.eval_batch_size
    if eval_batch_size < 1:
        raise ValueError("Evaluation batch size must be positive")
    if args.max_length > 512 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 \
            or not 0 <= args.warmup_ratio < 1 or not 0 < args.mask_prob < 1 or args.weight_decay < 0:
        raise ValueError("Invalid max length, learning rate, warmup ratio, mask probability or weight decay")
    output, checkpoint = args.output_dir, args.checkpoint
    if (output.exists() and any(output.iterdir())) or (
            checkpoint.exists() and any(p.name != ".gitkeep" for p in checkpoint.iterdir())):
        raise ValueError("Run/checkpoint directory already contains files; choose new paths")
    base_source = verify_base_model(args.base_model)
    code_hashes = {name: file_sha256(ROOT / name) for name in MLM_CODE_FILES}
    device = choose_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = False
    mlm_corpus = getattr(args, "mlm_corpus", "split-train")
    if mlm_corpus not in ("split-train", "all-dataset"):
        raise ValueError("--mlm-corpus must be split-train or all-dataset")
    train_rows, val_rows, manifest = load_split(args.data_dir, args.split)
    if mlm_corpus == "all-dataset":
        # every dataset text, no held-out text: no MLM eval and (downstream) no strict OOF
        corpus_ids = [r["id"] for r in train_rows + val_rows]
        if len(set(corpus_ids)) != len(corpus_ids):
            raise ValueError("all-dataset MLM corpus contains duplicate ids")
        train_rows, val_rows = train_rows + val_rows, []
        if args.smoke:
            train_rows = smoke_subset(train_rows, args.seed)
    elif args.smoke:
        train_rows, val_rows = [smoke_subset(part, args.seed) for part in (train_rows, val_rows)]
    epochs = 1 if args.smoke else args.epochs
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_source["directory"], local_files_only=True,
                                              trust_remote_code=False)
    model = AutoModelForMaskedLM.from_pretrained(base_source["directory"], local_files_only=True,
                                                 use_safetensors=True, trust_remote_code=False).to(device)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mask_prob)
    train_batches = collate_rows(train_rows, tokenizer, args.max_length, args.batch_size, collator,
                                 args.preprocessing, shuffle=True, seed=args.seed)
    val_batches = None if not val_rows else fixed_mask_batches(
        collate_rows(val_rows, tokenizer, args.max_length, eval_batch_size, collator, args.preprocessing),
        args.seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, foreach=False)
    steps_per_epoch = math.ceil(len(train_batches) / args.accumulation_steps)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    environment = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                   "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}
    config = {"model": base_source["model"], "revision": base_source["revision"],
              "base_model_source": base_source, "training_code_sha256": code_hashes,
              "seed": args.seed, "preprocessing": args.preprocessing, "max_length": args.max_length,
              "batch_size": args.batch_size, "eval_batch_size": eval_batch_size,
              "eval_batch_size_arg": getattr(args, "eval_batch_size", None), "accumulation_steps": args.accumulation_steps,
              "epochs": epochs, "learning_rate": args.learning_rate, "mask_prob": args.mask_prob,
              "masking": "80/10/10 DataCollatorForLanguageModeling, special tokens never masked",
              "warmup_ratio": args.warmup_ratio, "warmup_steps": warmup_steps, "weight_decay": args.weight_decay,
              "optimizer_steps_planned": total_steps, "amp": amp,
              "gradient_checkpointing": not args.no_gradient_checkpointing, "smoke": args.smoke,
              "model_selection": "none; final-epoch weights are saved",
              "mlm_corpus": mlm_corpus,
              "validation_use": ("fixed-mask MLM eval loss only (no gradient)" if mlm_corpus == "split-train"
                                 else "none; corpus is all dataset texts (no held-out text)"),
              "dataset_sha256": manifest["dataset_sha256"], "split_path": str(args.split),
              "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
              "mlm_text_ids": [r["id"] for r in train_rows], "validation_ids": [r["id"] for r in val_rows],
              "environment": environment}
    write_json(output / "run_config.json", config)
    write_json(output / "environment.json", environment)
    print(json.dumps(environment), flush=True)
    print(f"MLM corpus ({mlm_corpus}) {len(train_rows)} texts / {len(val_rows)} validation texts (eval only); "
          f"optimizer steps planned: {total_steps}", flush=True)
    history, update_count, attempted_updates, empty_windows = [], 0, 0, 0
    started = time.perf_counter()
    total_train_seconds, total_eval_seconds = 0.0, 0.0
    for epoch in range(1, epochs + 1):
        epoch_started = sync_cuda(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, token_count = 0.0, 0
        window_has_grad = False
        for index, batch in enumerate(train_batches):
            window = min(args.accumulation_steps, len(train_batches) - (index // args.accumulation_steps) * args.accumulation_steps)
            n = masked_count(batch)
            if n > 0:  # a tiny batch can draw no mask; its loss would be NaN, so it contributes nothing
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite MLM training loss")
                scaler.scale(loss / window).backward()
                window_has_grad = True
                loss_sum += loss.detach().float().item() * n
                token_count += n
            if (index + 1) % args.accumulation_steps == 0 or index + 1 == len(train_batches):
                if not window_has_grad:  # every batch in this window drew no mask: nothing to unscale or step
                    empty_windows += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                window_has_grad = False
                attempted_updates += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    update_count += 1
                optimizer.zero_grad(set_to_none=True)
                if args.smoke or update_count % 25 == 0:
                    print(f"Epoch {epoch}: batch {index+1}/{len(train_batches)}, updates {update_count}", flush=True)
        if token_count == 0:
            raise ValueError("No masked tokens were drawn during the epoch")
        eval_started = sync_cuda(device)
        validation_loss = evaluate_mlm(model, val_batches, device, amp) if val_batches else None
        eval_seconds = sync_cuda(device) - eval_started
        train_seconds = eval_started - epoch_started
        total_train_seconds += train_seconds
        total_eval_seconds += eval_seconds
        row = {"epoch": epoch, "train_mlm_loss": loss_sum / token_count,
               "validation_mlm_loss": validation_loss,
               "train_seconds": train_seconds, "eval_seconds": eval_seconds,
               "learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - epoch_started,
               "optimizer_updates": update_count, "attempted_updates": attempted_updates,
               "empty_windows": empty_windows}
        history.append(row)
        write_json(output / "history.json", history)
        plot_mlm_history(history, output / "curves.png")
        val_text = "n/a" if row["validation_mlm_loss"] is None else f"{row['validation_mlm_loss']:.4f}"
        print(f"Epoch {epoch}: train MLM={row['train_mlm_loss']:.4f}, val MLM={val_text}, "
              f"{row['seconds']:.1f}s", flush=True)
    if update_count == 0 or scheduler.last_epoch != update_count:
        raise ValueError("No optimizer updates succeeded or scheduler steps differ from updates")
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else None
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    write_json(checkpoint / "mlm_config.json", {**config, "final_epoch": epochs,
                                                "final_validation_mlm_loss": history[-1]["validation_mlm_loss"]})
    summary = {"smoke": args.smoke, "checkpoint": str(checkpoint.resolve()), "epochs": epochs,
               "final_train_mlm_loss": history[-1]["train_mlm_loss"],
               "final_validation_mlm_loss": history[-1]["validation_mlm_loss"],
               "optimizer_updates": update_count, "attempted_updates": attempted_updates,
               "skipped_updates": attempted_updates - update_count, "empty_windows": empty_windows, "peak_allocated_mb": peak,
               "peak_reserved_mb": peak_reserved, "total_train_seconds": total_train_seconds,
               "total_eval_seconds": total_eval_seconds,
               "total_seconds": time.perf_counter() - started,
               "model_sha256": file_sha256(checkpoint / "model.safetensors")}
    write_json(output / "summary.json", summary)
    shutil.copyfile(output / "summary.json", checkpoint / "mlm_summary.json")  # completion marker next to the weights
    del optimizer, scheduler, model
    gc.collect()
    print(json.dumps(summary, indent=2), flush=True)


def check_init_from(init_from, base_source, manifest, train_rows, val_rows, smoke=False, allow_val_overlap=False):
    """Fail closed (before any output directory exists) unless the MLM checkpoint is safe to fine-tune from.

    `allow_val_overlap=True` lets the MLM corpus overlap this run's validation ids (the corpus must still be a subset
    of train + validation ids); the record then says whether the run is a strict OOF (zero overlap) or not.

    Returns the `init_from` record stored in run_config / training_config.
    """
    path = Path(init_from)
    if not path.is_dir():
        raise ValueError(f"--init-from directory does not exist: {path}")
    for name in MLM_REQUIRED_FILES:
        if not (path / name).is_file():
            raise ValueError(f"--init-from is incomplete (missing {name}); the MLM run did not finish")
    mlm = json.loads((path / "mlm_config.json").read_text(encoding="utf-8"))
    missing = [k for k in MLM_REQUIRED_CONFIG_KEYS if k not in mlm]
    if missing:
        raise ValueError(f"mlm_config.json lacks fields: {missing}")
    if mlm["smoke"] and not smoke:
        raise ValueError("--init-from points to a smoke MLM checkpoint")
    if mlm["base_model_source"] != base_source:
        raise ValueError("MLM checkpoint was built from a different base model source")
    if mlm["dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("MLM checkpoint dataset hash differs from this run's split")
    text_ids = mlm["mlm_text_ids"]
    if len(set(text_ids)) != len(text_ids):
        raise ValueError("MLM text ids contain duplicates")
    val_ids = {r["id"] for r in val_rows}
    n_overlap = len(set(text_ids) & val_ids)
    if allow_val_overlap:
        if not set(text_ids) <= {r["id"] for r in train_rows} | val_ids:
            raise ValueError("MLM corpus contains ids outside this run's training and validation ids")
    else:
        if n_overlap:
            raise ValueError("MLM corpus overlaps this run's validation ids")
        if not set(text_ids) <= {r["id"] for r in train_rows}:
            raise ValueError("MLM corpus is not a subset of this run's training ids")
    return {"path": str(path), "model_sha256": file_sha256(path / "model.safetensors"),
            "mlm_run_config": {"epochs": mlm["epochs"], "learning_rate": mlm["learning_rate"],
                               "mask_prob": mlm["mask_prob"], "split_sha256": mlm["split_sha256"],
                               "n_texts": len(text_ids), "mlm_corpus": mlm.get("mlm_corpus", "split-train")},
            "validation_overlap": {"allowed": bool(allow_val_overlap), "n_overlap_ids": n_overlap,
                                   "strict_oof": n_overlap == 0}}


def load_encoder_from_mlm(model, init_from, base_tokenizer):
    """Copy embeddings + encoder weights of the MLM checkpoint into a freshly built sequence classifier.

    Pooler and classifier keep the values from the base-model load (pooler from bert-base-uncased, classifier
    seeded as usual), since BertForMaskedLM has no pooler.
    """
    tokenizer = AutoTokenizer.from_pretrained(init_from, local_files_only=True, trust_remote_code=False)
    if tokenizer.get_vocab() != base_tokenizer.get_vocab():
        raise ValueError("MLM checkpoint tokenizer vocabulary differs from the base tokenizer")
    mlm_model = AutoModelForMaskedLM.from_pretrained(init_from, local_files_only=True, use_safetensors=True,
                                                     trust_remote_code=False)
    model.bert.embeddings.load_state_dict(mlm_model.bert.embeddings.state_dict(), strict=True)
    model.bert.encoder.load_state_dict(mlm_model.bert.encoder.state_dict(), strict=True)
    del mlm_model
    gc.collect()
