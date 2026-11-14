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
    parser.add_argument("command", choices=["inspect", "eda", "prepare", "verify-base", "train", "predict"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split", type=Path, default=ROOT / "outputs/eda/split.json")
    parser.add_argument("--base-model", type=Path, default=ROOT / "outputs/base_model")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--class-weighting", choices=["none", "sqrt-inverse"], default="none")
    from src.preprocessing import MODES
    parser.add_argument("--preprocessing", choices=MODES, default="whitespace-v1")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Train one epoch on 16 examples per class per split")
    args = parser.parse_args()

    try:
        if args.command in ("inspect", "eda"):
            inspect_dataset(args.data_dir / "dataset.csv", {"id", "text", "label"})
            inspect_dataset(args.data_dir / "benchmark.csv", {"id", "text"})
        if args.command == "eda":
            from src.eda import run_eda
            run_eda(args.data_dir, args.output_dir or ROOT / "outputs/eda", args.seed, args.validation_fraction)
        elif args.command == "train":
            from src.training import train
            train(args)
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
