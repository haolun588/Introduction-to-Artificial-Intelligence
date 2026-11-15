"""Verify packaged files and, optionally, reproduce the registered CSV exactly."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, predictions=None):
    root = root.resolve()
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    for relative, item in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe package path: {relative}")
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Package file differs: {relative}")
    result = {"candidate": manifest["candidate"], "files_verified": len(manifest["files"])}
    if predictions is not None:
        def read(path, fields):
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != fields:
                    raise ValueError(f"Unexpected header: {path}")
                records = list(reader)
            if any(set(r) != set(fields) or any(v is None for v in r.values()) for r in records):
                raise ValueError(f"Malformed CSV: {path}")
            return records
        expected = read(root / "reference/predictions.csv", ["id", "pred"])
        actual = read(predictions, ["id", "pred"])
        benchmark = read(root / "dataset/benchmark.csv", ["id", "text"])
        ids = [r["id"] for r in benchmark]
        if len(actual) != 1000 or len(set(ids)) != 1000 or [r["id"] for r in actual] != ids:
            raise ValueError("Prediction count/ID order differs")
        if any(r["pred"] not in {"negative", "neutral", "positive"} for r in actual):
            raise ValueError("Invalid label")
        if actual != expected or digest(predictions) != manifest["reference_predictions_sha256"]:
            raise ValueError("Predictions differ from the registered reference")
        result.update({"prediction_rows": len(actual), "label_differences": 0,
                       "predictions_sha256": digest(predictions), "byte_identical": True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.predictions), indent=2))


if __name__ == "__main__":
    main()
