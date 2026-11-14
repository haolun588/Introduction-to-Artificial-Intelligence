"""Explicit PyTorch training/evaluation loops for the HW3 BERT baseline."""

import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .eda import LABELS, preprocess
from .preprocessing import MODES, prepare_text
from .model_source import file_sha256, verify_base_model

ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING = "whitespace-v1"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_rows(path, labeled=True):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"id", "text", "label"} if labeled else {"id", "text"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Unexpected columns in {path}")
        rows = list(reader)
    if not rows or any(not r["id"] or not (r["text"] or "").strip() for r in rows):
        raise ValueError("Expected nonempty IDs and texts")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs")
    if labeled and any(r["label"] not in LABELS for r in rows):
        raise ValueError("Unknown label")
    return rows


def load_split(data_dir, split_path):
    source = data_dir / "dataset.csv"
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Dataset hash differs from the fixed split")
    rows = read_rows(source)
    by_id = {r["id"]: r for r in rows}
    train_ids, val_ids = manifest["train_ids"], manifest["validation_ids"]
    if (len(set(train_ids + val_ids)) != len(train_ids + val_ids)
            or set(train_ids + val_ids) != set(by_id)):
        raise ValueError("Split must partition every dataset ID exactly once")
    train, val = [[by_id[i] for i in ids] for ids in (train_ids, val_ids)]
    if {preprocess(r['text']) for r in train} & {preprocess(r['text']) for r in val}:
        raise ValueError("Normalized text leakage across split")
    if any({r['label'] for r in part} != set(LABELS) for part in (train, val)):
        raise ValueError("Both splits must contain every class")
    return train, val, manifest


def smoke_subset(rows, seed, per_class=16):
    rng = random.Random(seed)
    selected = []
    for label in LABELS:
        group = [r for r in rows if r["label"] == label]
        selected.extend(rng.sample(group, min(per_class, len(group))))
    rng.shuffle(selected)
    return selected


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length, labeled=True, preprocessing=PREPROCESSING):
        self.rows = rows
        self.encoded = tokenizer([prepare_text(r["text"], preprocessing) for r in rows],
                                 truncation=True, max_length=max_length, padding=False)
        self.labeled = labeled

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.encoded.items()}
        if self.labeled:
            item["labels"] = LABELS.index(self.rows[index]["label"])
        return item


def loader(rows, tokenizer, max_length, batch_size, shuffle=False, seed=42, labeled=True, preprocessing=PREPROCESSING):
    from transformers import DataCollatorWithPadding
    return DataLoader(TextDataset(rows, tokenizer, max_length, labeled, preprocessing), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"))


def metrics(truth, predicted):
    report = classification_report(truth, predicted, labels=list(range(3)),
                                   target_names=list(LABELS), zero_division=0, output_dict=True)
    return {"macro_f1": report["macro avg"]["f1-score"], "classes": {k: report[k] for k in LABELS},
            "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(3))).tolist()}


def choose_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; install a CUDA PyTorch build")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


@torch.inference_mode()
def evaluate(model, batches, device, amp=False):
    model.eval()
    truth, predicted, loss_sum = [], [], 0.0
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(**batch).logits
        loss_sum += F.cross_entropy(logits.float(), labels, reduction="sum").item()
        truth.extend(labels.tolist())
        predicted.extend(logits.argmax(-1).tolist())
    return {"loss": loss_sum / len(truth), **metrics(truth, predicted)}


def accumulation_sample_count(batch_index, batch_size, accumulation, dataset_size):
    start = (batch_index // accumulation) * accumulation * batch_size
    return min(accumulation * batch_size, dataset_size - start)


def class_weights(rows, method):
    """Train-only inverse-square-root weights, with mean sample weight one."""
    if method == "none":
        return None
    if method != "sqrt-inverse":
        raise ValueError(f"Unknown class weighting: {method}")
    counts = torch.tensor([sum(r["label"] == label for r in rows) for label in LABELS], dtype=torch.float64)
    if (counts <= 0).any():
        raise ValueError("Class weighting requires every class in training data")
    weights = counts.rsqrt()
    weights /= (counts * weights).sum() / counts.sum()
    return weights.float()


def training_loss(logits, labels, weights, window_samples):
    # Fixed sample normalization preserves class weighting even for micro-batch 1.
    # Do not average separately by each micro-batch's sum of class weights.
    return F.cross_entropy(logits.float(), labels, weight=weights, reduction="sum") / window_samples


def plot_history(history, path):
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "macro_f1")):
        for split in ("train", "validation"):
            ax.plot([r["epoch"] for r in history], [r[split][metric] for r in history], "o-", label=split)
        ax.set(xlabel="Epoch", ylabel=metric, xticks=[r["epoch"] for r in history])
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(args):
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.max_length) < 1:
        raise ValueError("Epochs, batch size, accumulation and max length must be positive")
    if args.max_length > 512 or args.learning_rate <= 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("Invalid max length, learning rate or warmup ratio")
    output = args.output_dir or ROOT / "outputs" / ("smoke" if args.smoke else "baseline")
    checkpoint = args.checkpoint or (output / "checkpoint" if args.smoke else ROOT / "saved_models/checkpoint")
    if (output.exists() and any(output.iterdir())) or (checkpoint.exists() and any(p.name != ".gitkeep" for p in checkpoint.iterdir())):
        raise ValueError("Run/checkpoint directory already contains files; choose new paths")
    # Fail closed before touching CUDA, loading weights, or creating run directories.
    base_source = verify_base_model(args.base_model)
    code_hashes = {name: file_sha256(ROOT / name) for name in
                   ("main.py", "src/__init__.py", "src/training.py", "src/eda.py",
                    "src/preprocessing.py", "src/model_source.py")}
    device = choose_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = False
    train_rows, val_rows, manifest = load_split(args.data_dir, args.split)
    if args.smoke:
        train_rows, val_rows = [smoke_subset(part, args.seed) for part in (train_rows, val_rows)]
    weights = class_weights(train_rows, args.class_weighting)
    if weights is not None:
        weights = weights.to(device)
    epochs = 1 if args.smoke else args.epochs
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_source["directory"], local_files_only=True,
                                              trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_source["directory"], local_files_only=True, use_safetensors=True,
        trust_remote_code=False, num_labels=3,
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
    ).to(device)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_batches = loader(train_rows, tokenizer, args.max_length, args.batch_size, True, args.seed, preprocessing=args.preprocessing)
    train_eval = loader(train_rows, tokenizer, args.max_length, args.batch_size, preprocessing=args.preprocessing)
    val_batches = loader(val_rows, tokenizer, args.max_length, args.batch_size, preprocessing=args.preprocessing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01, foreach=False)
    total_steps = epochs * math.ceil(len(train_batches) / args.accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    config = {"model": base_source["model"], "revision": base_source["revision"],
              "base_model_source": base_source, "training_code_sha256": code_hashes,
              "seed": args.seed, "labels": list(LABELS),
              "preprocessing": args.preprocessing, "max_length": args.max_length, "batch_size": args.batch_size,
              "accumulation_steps": args.accumulation_steps, "learning_rate": args.learning_rate,
              "weight_decay": 0.01, "warmup_ratio": args.warmup_ratio, "epochs": epochs,
              "class_weighting": args.class_weighting,
              "class_weights": weights.tolist() if weights is not None else None,
              "loss_normalization": "weighted sum / actual accumulation-window sample count; train mean weight 1",
              "evaluation_loss": "unweighted cross entropy for baseline comparison",
              "amp": amp, "gradient_checkpointing": not args.no_gradient_checkpointing,
              "smoke": args.smoke, "dataset_sha256": manifest["dataset_sha256"],
              "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
              "train_ids": [r["id"] for r in train_rows], "validation_ids": [r["id"] for r in val_rows]}
    write_json(output / "run_config.json", config)
    write_json(output / "split.json", manifest)
    environment = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                   "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}
    write_json(output / "environment.json", environment)
    print(json.dumps(environment), flush=True)
    print(f"Train {len(train_rows)} / validation {len(val_rows)}; optimizer steps planned: {total_steps}", flush=True)
    print(f"Class weights ({', '.join(LABELS)}): {config['class_weights']}", flush=True)
    best_f1, history, update_count = -1.0, [], 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(train_batches):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(**batch).logits
                # Normalize by actual samples, including the final partial accumulation window.
                loss = training_loss(logits, labels, weights, accumulation_sample_count(
                    index, args.batch_size, args.accumulation_steps, len(train_rows)))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            scaler.scale(loss).backward()
            if (index + 1) % args.accumulation_steps == 0 or index + 1 == len(train_batches):
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
        row = {"epoch": epoch, "train": evaluate(model, train_eval, device, amp),
               "validation": evaluate(model, val_batches, device, amp),
               "seconds": time.perf_counter() - epoch_started, "optimizer_updates": update_count}
        history.append(row)
        if row["validation"]["macro_f1"] > best_f1:
            best_f1 = row["validation"]["macro_f1"]
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            write_json(checkpoint / "training_config.json", {**config, "best_epoch": epoch, "validation": row["validation"]})
        write_json(output / "history.json", history)
        plot_history(history, output / "curves.png")
        print(f"Epoch {epoch}: train F1={row['train']['macro_f1']:.4f}, val F1={row['validation']['macro_f1']:.4f}, {row['seconds']:.1f}s", flush=True)
    if update_count == 0:
        raise ValueError("No optimizer updates succeeded")
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    # Release optimizer state before loading the selected checkpoint on a small GPU.
    del optimizer, scheduler, model, batch, logits, loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded, _, saved = load_checkpoint(checkpoint, device)
    reloaded_metrics = evaluate(reloaded, val_batches, device, amp)
    expected = saved["validation"]
    if not math.isclose(reloaded_metrics["loss"], expected["loss"], rel_tol=1e-5, abs_tol=1e-6) or reloaded_metrics["confusion_matrix"] != expected["confusion_matrix"]:
        raise ValueError("Reloaded checkpoint does not reproduce selected validation metrics")
    summary = {"smoke": args.smoke, "checkpoint": str(checkpoint.resolve()), "best_epoch": saved["best_epoch"],
               "best_validation_macro_f1": best_f1, "optimizer_updates": update_count,
               "peak_allocated_mb": peak, "total_seconds": time.perf_counter() - started,
               "reload_verified": True, "reloaded_validation": reloaded_metrics}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def load_checkpoint(path, device):
    config = json.loads((path / "training_config.json").read_text(encoding="utf-8"))
    if config["labels"] != list(LABELS) or config["preprocessing"] not in MODES:
        raise ValueError("Unsupported checkpoint label mapping or preprocessing")
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(device)
    if model.config.id2label != dict(enumerate(LABELS)):
        raise ValueError("Model label mapping differs from training metadata")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model.eval()
    return model, tokenizer, config


@torch.inference_mode()
def predict(args):
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    checkpoint = args.checkpoint or ROOT / "saved_models/checkpoint"
    destination = args.predictions or ROOT / "outputs/predictions.csv"
    if destination.exists():
        raise ValueError("Prediction file already exists; choose a new --predictions path")
    device = choose_device(args.device)
    model, tokenizer, config = load_checkpoint(checkpoint, device)
    rows = read_rows(args.data_dir / "benchmark.csv", labeled=False)
    batches = loader(rows, tokenizer, config["max_length"], args.batch_size, labeled=False, preprocessing=config["preprocessing"])
    predictions = []
    for batch in batches:
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        predictions.extend(LABELS[i] for i in logits.argmax(-1).tolist())
    if len(predictions) != len(rows):
        raise ValueError("Prediction count mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "pred"])
        writer.writeheader()
        writer.writerows({"id": row["id"], "pred": label} for row, label in zip(rows, predictions))
    print(f"Wrote {len(rows)} predictions in source ID order: {destination}")
    if config["smoke"]:
        print("SMOKE checkpoint: pipeline test only, not a trained baseline for submission.")
