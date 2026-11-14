"""Compare fixed-validation errors and equal-weight soft voting of two checkpoints."""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.training import (ROOT, LABELS, choose_device, load_checkpoint, load_split,
                      loader, metrics, read_rows, write_json)


def write_csv(path, rows, fieldnames=None):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def probabilities(checkpoint, rows, device, expected_ids, split_hash, dataset_hash):
    model, tokenizer, config = load_checkpoint(checkpoint, device)
    if (config['validation_ids'] != expected_ids or config['split_sha256'] != split_hash
            or config['dataset_sha256'] != dataset_hash or config['smoke']):
        raise ValueError('Checkpoint does not match the full fixed validation split')
    batches = loader(rows, tokenizer, config['max_length'], 2, labeled=False,
                     preprocessing=config['preprocessing'])
    values = []
    for batch in batches:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda' and config['amp']):
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        values.append(logits.float().softmax(-1).cpu().numpy())
    result = np.concatenate(values)
    if result.shape != (len(rows), len(LABELS)) or not np.isfinite(result).all():
        raise ValueError('Invalid probability matrix')
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)
    del model, tokenizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'auto'])
    parser.add_argument('--pair', choices=['learning-rate', 'seed'], default='learning-rate')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--generate-benchmark', action='store_true',
                        help='Generate an experimental submission even if validation does not improve')
    args = parser.parse_args()
    output = args.output_dir or ROOT / ('outputs/ensemble_seed42_seed43' if args.pair == 'seed'
                                      else 'outputs/ensemble_lr2_lr3')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; use a new path')
    output.mkdir(parents=True, exist_ok=True)
    split = ROOT / 'outputs/eda/split.json'
    _, rows, _ = load_split(ROOT / 'dataset', split)
    ids = [r['id'] for r in rows]
    split_hash = hashlib.sha256(split.read_bytes()).hexdigest()
    dataset_hash = hashlib.sha256((ROOT / 'dataset/dataset.csv').read_bytes()).hexdigest()
    device = choose_device(args.device)
    checkpoints = [ROOT / 'outputs/weighted_sqrt/checkpoint', ROOT / 'outputs/weighted_lr3e5/checkpoint']
    names = ['lr2e5', 'lr3e5']
    if args.pair == 'seed':
        checkpoints[1] = ROOT / 'outputs/weighted_seed43/checkpoint'
        names = ['seed42', 'seed43']
    truth = np.array([LABELS.index(r['label']) for r in rows])
    matrices, results = [], {}
    for name, checkpoint in zip(names, checkpoints):
        matrix, config = probabilities(checkpoint, rows, device, ids, split_hash, dataset_hash)
        result = metrics(truth, matrix.argmax(axis=1))
        if result['confusion_matrix'] != config['validation']['confusion_matrix']:
            raise ValueError('Single-model validation does not reproduce saved confusion matrix')
        matrices.append(matrix)
        results[name] = result
        print(f'{name}: {result["macro_f1"]:.6f}; saved validation reproduced', flush=True)
    averaged = (matrices[0] + matrices[1]) / 2
    predictions = [m.argmax(axis=1) for m in [*matrices, averaged]]
    results['soft_voting'] = metrics(truth, predictions[2])
    correct = [p == truth for p in predictions]
    overlap = {'both_correct': int((correct[0] & correct[1]).sum()),
               f'only_{names[0]}_correct': int((correct[0] & ~correct[1]).sum()),
               f'only_{names[1]}_correct': int((~correct[0] & correct[1]).sum()),
               'both_wrong': int((~correct[0] & ~correct[1]).sum()),
               'disagreements': int((predictions[0] != predictions[1]).sum()),
               f'ensemble_fixes_{names[0]}': int((~correct[0] & correct[2]).sum()),
               f'ensemble_breaks_{names[0]}': int((correct[0] & ~correct[2]).sum())}
    records = []
    for i, row in enumerate(rows):
        record = dict(row)
        record['error_group'] = ('both_correct' if correct[0][i] and correct[1][i] else
                                 f'only_{names[0]}_correct' if correct[0][i] else
                                 f'only_{names[1]}_correct' if correct[1][i] else 'both_wrong')
        for name, matrix, pred in zip([*names, 'soft_voting'], [*matrices, averaged], predictions):
            record[f'{name}_pred'] = LABELS[pred[i]]
            for j, label in enumerate(LABELS):
                record[f'{name}_p_{label}'] = float(matrix[i, j])
        records.append(record)
    write_csv(output / 'validation_predictions.csv', records)
    write_csv(output / 'validation_errors.csv', [r for i, r in enumerate(records)
                                                if not all(c[i] for c in correct)], list(records[0]))
    selected = results['soft_voting']['macro_f1'] > max(results[n]['macro_f1'] for n in names)
    summary = {'labels': list(LABELS), 'checkpoints': [str(p.resolve()) for p in checkpoints],
               'names': names, 'pair': args.pair,
               'weights': [0.5, 0.5], 'tie_break': 'first class in labels order (argmax)',
               'validation_ids': ids, 'split_sha256': split_hash, 'dataset_sha256': dataset_hash,
               'metrics': results, 'overlap': overlap, 'ensemble_selected': selected,
               'benchmark_generated': False, 'benchmark_requested': args.generate_benchmark,
               'device': str(device),
               'precision': 'checkpoint AMP setting on CUDA; float32 softmax and averaging'}
    write_json(output / 'summary.json', summary)
    if selected or args.generate_benchmark:
        benchmark = read_rows(ROOT / 'dataset/benchmark.csv', labeled=False)
        benchmark_matrices = [probabilities(p, benchmark, device, ids, split_hash, dataset_hash)[0]
                              for p in checkpoints]
        labels = ((benchmark_matrices[0] + benchmark_matrices[1]) / 2).argmax(axis=1)
        write_csv(output / 'predictions.csv', [{'id': r['id'], 'pred': LABELS[v]}
                                              for r, v in zip(benchmark, labels)])
        summary['benchmark_generated'] = True
        write_json(output / 'summary.json', summary)
    print(json.dumps({'metrics': results, 'overlap': overlap, 'ensemble_selected': selected}, indent=2))


if __name__ == '__main__':
    main()
