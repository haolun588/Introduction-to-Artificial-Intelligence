"""HW3 workspace entry point."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {"negative", "neutral", "positive"}


def inspect_dataset(path: Path, expected_fields: set[str]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if set(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"{path.name}: unexpected CSV header {reader.fieldnames}")
        rows = list(reader)

    ids = [row["id"] for row in rows]
    if not rows or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: expected nonempty, unique IDs")

    texts = [row["text"] or "" for row in rows]
    print(f"{path.name}: {len(rows):,} rows")
    print(f"  Blank texts: {sum(not value.strip() for value in texts):,}")
    print(f"  Extra rows with identical text: {len(texts) - len(set(texts)):,}")

    if "label" in expected_fields:
        counts = Counter(row["label"] for row in rows)
        if set(counts) != LABELS:
            raise ValueError(f"{path.name}: expected labels {sorted(LABELS)}")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count:,} ({count / len(rows):.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "eda", "prepare", "verify-base", "train", "predict", "kfold", "pretrain-mlm"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5, help="Number of CV folds for the kfold command")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split", type=Path, default=None,
                        help="fold/split JSON (default outputs/eda/split.json; required for pretrain-mlm)")
    parser.add_argument("--base-model", type=Path, default=ROOT / "outputs/base_model")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=None, help="train default 3; pretrain-mlm default 10")
    parser.add_argument("--eval-interval", type=float, choices=[0.5, 1.0], default=1.0,
                        help="Evaluate every half or full epoch, aligned to accumulation boundaries")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Batch size of evaluation loaders (train/pretrain-mlm); default None reuses --batch-size")
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=None, help="train default 2e-5; pretrain-mlm default 5e-5")
    parser.add_argument("--layer-lr-decay", type=float, default=1.0,
                        help="BERT learning-rate multiplier per lower layer; 1 keeps uniform LR")
    parser.add_argument("--class-weighting", choices=["none", "sqrt-inverse", "quarter-inverse"], default="none")
    parser.add_argument("--fgsm-epsilon", type=float, default=0.0,
                        help="Word-embedding FGSM L-infinity radius; 0 disables adversarial training")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon in [0, 1) for the training loss only (per-sample, then class-weighted)")
    parser.add_argument("--exclude-train-ids", type=Path, default=None,
                        help="Text file, one id per line: remove these ids from the training subset only (validation unchanged)")
    from src.preprocessing import MODES
    parser.add_argument("--preprocessing", choices=MODES, default="whitespace-v1")
    parser.add_argument("--warmup-ratio", type=float, default=None, help="train default 0.1; pretrain-mlm default 0.06")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="pretrain-mlm: masking probability")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="pretrain-mlm: AdamW weight decay (train fixes 0.01)")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="train: MLM checkpoint from pretrain-mlm; encoder weights are reused, classifier is newly initialized")
    parser.add_argument("--mlm-corpus", choices=["split-train", "all-dataset"], default="split-train",
                        help="pretrain-mlm: split-train (default; the split's training texts) or all-dataset "
                             "(every dataset text, no held-out text, downstream OOF not strict)")
    parser.add_argument("--allow-mlm-val-overlap", action="store_true",
                        help="train: allow the --init-from MLM corpus to overlap validation ids (recorded as non-strict OOF)")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Train one epoch on 16 examples per class per split")
    args = parser.parse_args()
    mlm_defaults = args.command == "pretrain-mlm"
    if args.split is None and not mlm_defaults:
        args.split = ROOT / "outputs/eda/split.json"
    for name, train_default, mlm_default in (("epochs", 3, 10), ("learning_rate", 2e-5, 5e-5), ("warmup_ratio", 0.1, 0.06)):
        if getattr(args, name) is None:
            setattr(args, name, mlm_default if mlm_defaults else train_default)

    try:
        if args.command in ("inspect", "eda"):
            inspect_dataset(args.data_dir / "dataset.csv", {"id", "text", "label"})
            inspect_dataset(args.data_dir / "benchmark.csv", {"id", "text"})
        if args.command == "eda":
            from src.eda import run_eda
            run_eda(args.data_dir, args.output_dir or ROOT / "outputs/eda", args.seed, args.validation_fraction)
        elif args.command == "kfold":
            if not args.output_dir:
                raise ValueError("kfold requires --output-dir")
            from src.eda import run_kfold
            for entry in run_kfold(args.data_dir, args.output_dir, args.k, args.seed):
                print(entry["file"], entry["class_counts"]["validation"])
        elif args.command == "train":
            from src.training import train
            train(args)
        elif args.command == "pretrain-mlm":
            from src.mlm import pretrain_mlm
            pretrain_mlm(args)
        elif args.command == "predict":
            from src.training import predict
            predict(args)
        elif args.command == "verify-base":
            from src.model_source import verify_base_model
            print(json.dumps(verify_base_model(args.base_model), indent=2))
        elif args.command == "prepare":
            from huggingface_hub import snapshot_download
            from src.model_source import MODEL, REVISION, APPROVED_FILES, verify_base_model
            snapshot_download(MODEL, revision=REVISION, local_dir=args.base_model,
                              allow_patterns=list(APPROVED_FILES))
            verified = verify_base_model(args.base_model)
            print(f"Base model verified: {verified['directory']}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
