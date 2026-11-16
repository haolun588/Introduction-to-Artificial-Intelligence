"""Reproducible grouped split and train-only BERT EDA."""

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .model_source import MODEL, REVISION

LABELS = ("negative", "neutral", "positive")
PATTERNS = {
    "url": r"https?://|www\.",
    "hashtag": r"#\w+",
    "mention": r"@\w+",
    "leading_mention": r"^\s*@\w+",
    # Approximate Unicode ranges; not a complete emoji grapheme detector.
    "emoji_approx": r"[\U0001F000-\U0001FAFF\u2600-\u27BF]",
}


def preprocess(text):
    return " ".join(text.split())


def _groups_by_label(rows):
    """Whitespace-normalized text groups per label; identical text with different labels raises."""
    groups = defaultdict(list)
    for row in rows:
        groups[preprocess(row["text"])].append(row)
    by_label = defaultdict(list)
    for key, group in sorted(groups.items()):
        labels = {row["label"] for row in group}
        if len(labels) != 1:
            raise ValueError("Conflicting labels for identical normalized text")
        by_label[next(iter(labels))].append((key, group))
    return by_label


def grouped_kfold(rows, k=5, seed=42):
    """Return k validation-id sets; same normalized text always lands in one fold.

    Per label: shuffle groups with Random(seed), sort by size descending (stable, so equal
    sizes keep the shuffled order), then give each group to the fold with the fewest rows of
    that label (ties -> lowest fold index).
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    by_label = _groups_by_label(rows)
    rng = random.Random(seed)
    folds = [set() for _ in range(k)]
    for label in LABELS:
        candidates = by_label[label]
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: -len(item[1]))
        sizes = [0] * k
        for _, group in candidates:
            target = min(range(k), key=lambda i: (sizes[i], i))
            folds[target].update(row["id"] for row in group)
            sizes[target] += len(group)
    label_of = {row["id"]: row["label"] for row in rows}
    for fold in folds:
        if {label_of[i] for i in fold} != set(LABELS):
            raise ValueError("Each fold must contain all three labels")
    return folds


def run_kfold(data_dir: Path, output_dir: Path, k: int, seed: int):
    source = data_dir / "dataset.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(not r["text"].strip() for r in rows):
        raise ValueError("Blank training text")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory already exists and is not empty")
    folds = grouped_kfold(rows, k, seed)
    dataset_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, validation_ids in enumerate(folds):
        train = [r for r in rows if r["id"] not in validation_ids]
        validation = [r for r in rows if r["id"] in validation_ids]
        manifest = {
            "dataset_sha256": dataset_sha, "seed": seed,
            "algorithm": "per-label seeded greedy grouped kfold v1",
            "group_key": "whitespace-normalized text", "fold": i, "k": k,
            "train_ids": [r["id"] for r in train],
            "validation_ids": [r["id"] for r in validation],
            "class_counts": {name: dict(Counter(r["label"] for r in part))
                             for name, part in (("train", train), ("validation", validation))},
        }
        path = output_dir / f"fold{i}.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        entries.append({"fold": i, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "class_counts": manifest["class_counts"]})
    (output_dir / "folds_manifest.json").write_text(json.dumps(
        {"k": k, "seed": seed, "dataset_sha256": dataset_sha,
         "algorithm": "per-label seeded greedy grouped kfold v1", "folds": entries}, indent=2), encoding="utf-8")
    return entries


def grouped_split(rows, seed=42, validation_fraction=0.1):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    by_label = _groups_by_label(rows)
    rng = random.Random(seed)
    validation_keys = set()
    for label in LABELS:
        candidates = by_label[label]
        rng.shuffle(candidates)
        target = round(sum(len(group) for _, group in candidates) * validation_fraction)
        count = 0
        for key, group in candidates:
            if abs(count + len(group) - target) < abs(count - target):
                validation_keys.add(key)
                count += len(group)
    train = [r for r in rows if preprocess(r["text"]) not in validation_keys]
    validation = [r for r in rows if preprocess(r["text"]) in validation_keys]
    if any(set(r["label"] for r in part) != set(LABELS) for part in (train, validation)):
        raise ValueError("Each split must contain all three labels")
    return train, validation


def run_eda(data_dir: Path, output_dir: Path, seed: int, validation_fraction: float):
    from transformers import AutoTokenizer, __version__

    source = data_dir / "dataset.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(not r["text"].strip() for r in rows):
        raise ValueError("Blank training text")
    train, validation = grouped_split(rows, seed, validation_fraction)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": seed, "validation_fraction": validation_fraction,
        "algorithm": "per-label seeded greedy grouped split v1",
        "group_key": "whitespace-normalized text",
        "train_ids": [r["id"] for r in train],
        "validation_ids": [r["id"] for r in validation],
        "class_counts": {name: dict(Counter(r["label"] for r in part))
                         for name, part in (("train", train), ("validation", validation))},
    }
    manifest_path = output_dir / "split.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing split differs; use a different --output-dir to preserve experiments")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, part in (("train", train), ("validation", validation)):
        with (output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "text", "label"])
            writer.writeheader()
            writer.writerows(part)

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir) if (tokenizer_dir / "tokenizer_config.json").exists() else MODEL,
        cache_dir=str(output_dir / "hf_cache"),
        revision=REVISION,
    )
    tokenizer.save_pretrained(tokenizer_dir)
    texts = [preprocess(r["text"]) for r in train]
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
    lengths = sorted(map(len, encoded))
    percentiles = {f"P{p}": lengths[math.ceil(len(lengths) * p / 100) - 1] for p in (50, 90, 95, 99, 100)}
    truncation = {str(n): sum(length > n for length in lengths) / len(lengths)
                  for n in (32, 48, 64, 96, 128)}
    recommended = next((int(n) for n, rate in truncation.items() if rate <= 0.01), None)
    features = {}
    samples = {}
    for feature, pattern in PATTERNS.items():
        matches = [r for r in train if re.search(pattern, r["text"])]
        features[feature] = {
            label: {"count": sum(r["label"] == label for r in matches),
                    "total": manifest["class_counts"]["train"][label]}
            for label in LABELS
        }
        samples[feature] = [{"id": r["id"], "label": r["label"], "text": r["text"],
                             "tokens": tokenizer.tokenize(preprocess(r["text"]))}
                            for r in matches[:3]]
    mentions = Counter(m.lower() for text in texts for m in re.findall(r"@\w+", text))
    report = {
        "model": MODEL, "transformers_version": __version__,
        "tokenizer_revision": REVISION,
        "preprocessing": "whitespace normalization only",
        "scope": "training split only; lengths include CLS and SEP",
        "class_counts": manifest["class_counts"], "length_percentiles": percentiles,
        "truncation_rates": truncation, "recommended_max_length": recommended,
        "selection_rule": "shortest candidate with at most 1% truncated training rows",
        "rows_with_unk": sum(tokenizer.unk_token_id in ids for ids in encoded),
        "features": features, "top_mentions": mentions.most_common(15),
        "tokenization_samples": samples,
        "placeholder_tokens": {s: tokenizer.tokenize(s) for s in ("[URL]", "http", "@user")},
    }
    (output_dir / "eda.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# HW3 EDA", "", f"Seed: {seed}; train: {len(train)}; validation: {len(validation)}.",
             "Grouped by whitespace-normalized text; no shared groups across splits.", "",
             "All feature and token statistics below use training rows only.", "",
             "| Label | Train | Validation |", "|---|---:|---:|"]
    for label in LABELS:
        lines.append(f"| {label} | {manifest['class_counts']['train'][label]} | {manifest['class_counts']['validation'][label]} |")
    lines += ["", f"Token length (including CLS/SEP): {percentiles}", "",
              "| Max length | Truncated rows |", "|---|---:|"]
    lines += [f"| {n} | {rate:.2%} |" for n, rate in truncation.items()]
    lines += ["", f"Suggested max_length: {recommended}; use dynamic padding during training.",
              f"Rows containing UNK: {report['rows_with_unk']} / {len(train)}.", "",
              "| Feature | negative | neutral | positive |", "|---|---:|---:|---:|"]
    for feature, counts in features.items():
        cells = [f"{counts[label]['count']} ({counts[label]['count']/counts[label]['total']:.2%})" for label in LABELS]
        lines.append(f"| {feature} | " + " | ".join(cells) + " |")
    lines += ["", "Emoji detection uses approximate Unicode ranges, not full grapheme matching.",
              "", f"Top mentions: {mentions.most_common(10)}", "",
              "Keep baseline text unchanged except whitespace normalization. Evaluate URL/mention/emoji changes as separate experiments.",
              "Tokenization examples and placeholder token behavior: see eda.json.", "",
              "Tokenizer API: https://huggingface.co/docs/transformers/main_classes/tokenizer"]
    (output_dir / "EDA_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"EDA complete: {output_dir / 'EDA_Report.md'}")
    print(f"Train / validation: {len(train)} / {len(validation)}; recommended max_length: {recommended}")
