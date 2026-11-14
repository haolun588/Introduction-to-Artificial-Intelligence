"""Train-only audit of approximate emoji code points and BERT tokenization."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from transformers import AutoTokenizer

from src.eda import preprocess

ROOT = Path(__file__).resolve().parents[1]
EMOJI = re.compile(r'[\U0001F000-\U0001FAFF\u2600-\u27BF]')


def describe(text):
    # Diagnostic candidate only: Unicode character names, not a semantic emoji lexicon.
    text = EMOJI.sub(lambda m: ' ' + unicodedata.name(m[0], 'unknown symbol').lower() + ' ', text)
    return preprocess(text.replace('\ufe0f', '').replace('\ufe0e', ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/emoji_audit')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Output directory is not empty; choose a new path')
    split = json.loads((ROOT / 'outputs/eda/split.json').read_text(encoding='utf-8'))
    data = ROOT / 'dataset/dataset.csv'
    if hashlib.sha256(data.read_bytes()).hexdigest() != split['dataset_sha256']:
        raise ValueError('Dataset differs from fixed split')
    with data.open(encoding='utf-8-sig', newline='') as stream:
        by_id = {r['id']: r for r in csv.DictReader(stream)}
    rows = [by_id[i] for i in split['train_ids']]
    assert not set(split['train_ids']) & set(split['validation_ids'])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'outputs/base_model', local_files_only=True)
    totals, affected, chars = Counter(), Counter(), Counter()
    char_rows, examples, lengths_before, lengths_after = {}, [], [], []
    unknown_before = unknown_after = affected_unknown_before = 0
    for row in rows:
        text = preprocess(row['text'])
        found = EMOJI.findall(text)
        totals[row['label']] += 1
        tokens = tokenizer.tokenize(text)
        converted = describe(text) if found else text
        after = tokenizer.tokenize(converted)
        lengths_before.append(len(tokens) + 2)
        lengths_after.append(len(after) + 2)
        unknown_before += tokens.count(tokenizer.unk_token)
        unknown_after += after.count(tokenizer.unk_token)
        if not found:
            continue
        affected[row['label']] += 1
        affected_unknown_before += int(tokenizer.unk_token in tokens)
        chars.update(found)
        for char in set(found):
            char_rows.setdefault(char, Counter())[row['label']] += 1
        examples.append({'id': row['id'], 'label': row['label'], 'text': text,
                         'tokens': tokens, 'candidate_text': converted, 'candidate_tokens': after})
    inventory = []
    for char, count in chars.most_common():
        tokens = tokenizer.tokenize(char)
        inventory.append({'character': char, 'codepoint': f'U+{ord(char):04X}',
                          'unicode_name': unicodedata.name(char, 'UNNAMED'),
                          'occurrences': count, 'rows_by_label': dict(char_rows[char]),
                          'tokens': tokens, 'contains_unk': tokenizer.unk_token in tokens,
                          'description_tokens': tokenizer.tokenize(describe(char))})
    result = {'scope': 'fixed training split only; validation and benchmark excluded',
              'limitation': 'approximate codepoint ranges, not complete emoji or grapheme detection; Unicode names are not a sentiment lexicon',
              'dataset_sha256': split['dataset_sha256'], 'train_ids': split['train_ids'],
              'train_rows': len(rows), 'class_counts': dict(totals),
              'affected_rows': len(examples), 'affected_by_label': dict(affected),
              'unique_codepoints': len(chars), 'occurrences': sum(chars.values()),
              'standalone_unk_occurrences': sum(x['occurrences'] for x in inventory if x['contains_unk']),
              'affected_rows_with_any_unk': affected_unknown_before,
              'all_train_unk_tokens_before': unknown_before, 'all_train_unk_tokens_candidate': unknown_after,
              'rows_over_64_before': sum(n > 64 for n in lengths_before),
              'rows_over_64_candidate': sum(n > 64 for n in lengths_after),
              'max_length_before': max(lengths_before), 'max_length_candidate': max(lengths_after),
              'inventory': inventory}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [('summary.json', result), ('examples.json', examples)]:
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['train_ids','inventory']}, indent=2))
    print(json.dumps(inventory[:20], ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
