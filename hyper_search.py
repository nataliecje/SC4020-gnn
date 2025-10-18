"""Hyperparameter search script for SC4020-gnn

Usage examples:
    python3 scripts/hyper_search.py --mode random --trials 30 --models GCN GAT GraphSAGE GraphTransformer

This script will import ExperimentPipeline/ExperimentRunner-like entrypoints from `simple_pipeline.py` and run experiments with provided hyperparameters.

Notes:
- No weight decay (per user request).
- Two modes: grid, random. (Optuna can be added if requested.)
- Results are appended to results/hyper_search_<timestamp>.json and .csv
"""
import argparse
import json
import os
import random
import time
from itertools import product

import numpy as np

# We import the local pipeline
from simple_pipeline import ExperimentPipeline

# Ranges (user-approved)
DEPTHS = [1, 2, 3, 4]
HIDDENS = [16, 32, 64, 128]
DROPOUTS = [0.0, 0.2, 0.5]
LRS = [5e-4, 1e-3, 5e-3, 1e-2]
GAT_HEADS = [1, 2, 4]

RESULTS_DIR = 'results'


def grid_configs(models, fix_depth=None):
    configs = []
    depths = [fix_depth] if fix_depth is not None else DEPTHS
    for model in models:
        if model == 'GAT':
            for depth, hidden, dropout, lr, heads in product(depths, HIDDENS, DROPOUTS, LRS, GAT_HEADS):
                cfg = dict(model=model, num_layers=depth, hidden_channels=hidden, dropout=dropout, lr=lr, heads=heads)
                configs.append(cfg)
        else:
            for depth, hidden, dropout, lr in product(depths, HIDDENS, DROPOUTS, LRS):
                cfg = dict(model=model, num_layers=depth, hidden_channels=hidden, dropout=dropout, lr=lr)
                configs.append(cfg)
    return configs


def random_configs(models, trials=30, fix_depth=3, seed=None):
    rng = random.Random(seed)
    configs = []
    for _ in range(trials):
        for model in models:
            depth = fix_depth if fix_depth is not None else rng.choice(DEPTHS)
            hidden = rng.choice(HIDDENS)
            dropout = rng.choice(DROPOUTS)
            lr = float(10 ** rng.uniform(np.log10(min(LRS)), np.log10(max(LRS))))
            if model == 'GAT':
                heads = rng.choice(GAT_HEADS)
                cfg = dict(model=model, num_layers=depth, hidden_channels=hidden, dropout=dropout, lr=lr, heads=heads)
            else:
                cfg = dict(model=model, num_layers=depth, hidden_channels=hidden, dropout=dropout, lr=lr)
            configs.append(cfg)
    return configs


def run_config(pipeline, cfg):
    model = cfg.pop('model')
    print(f"Running {model} with {cfg}")
    res = pipeline.run_single_experiment(model, hidden_channels=cfg.get('hidden_channels'), num_layers=cfg.get('num_layers'), dropout=cfg.get('dropout'), lr=cfg.get('lr'), heads=cfg.get('heads', None))
    # Merge config into result
    if res is None:
        return None
    out = {
        'config': cfg,
        'model': model,
        'micro_f1': res['test_micro_f1'],
        'macro_f1': res['test_macro_f1'],
        'accuracy': res['test_accuracy'],
        'train_losses': res.get('train_losses', []),
        'val_f1_scores': res.get('val_f1_scores', [])
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['grid', 'random'], default='random')
    parser.add_argument('--trials', type=int, default=30, help='trials per model for random mode')
    parser.add_argument('--models', nargs='+', default=['GCN', 'GAT', 'GraphSAGE', 'GraphTransformer'])
    parser.add_argument('--fix-depth', type=int, default=3, help='If set, use this depth only (recommended)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_json = os.path.join(RESULTS_DIR, f'hyper_search_{timestamp}.json')
    out_csv = os.path.join(RESULTS_DIR, f'hyper_search_{timestamp}.csv')

    pipeline = ExperimentPipeline()

    if args.mode == 'grid':
        configs = grid_configs(args.models, fix_depth=args.fix_depth)
    else:
        configs = random_configs(args.models, trials=args.trials, fix_depth=args.fix_depth, seed=args.seed)

    print(f"Running {len(configs)} experiments...")

    results = []
    for cfg in configs:
        res = run_config(pipeline, cfg.copy())
        if res is not None:
            results.append(res)
            # append to JSON progressively
            with open(out_json, 'w') as f:
                json.dump(results, f, indent=2)

    # write CSV summary
    import csv
    if results:
        keys = ['model'] + sorted(list(results[0]['config'].keys())) + ['micro_f1', 'macro_f1', 'accuracy']
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for r in results:
                row = [r['model']] + [r['config'].get(k) for k in sorted(list(r['config'].keys()))] + [r['micro_f1'], r['macro_f1'], r['accuracy']]
                writer.writerow(row)

    print('Done. Results written to', out_json, out_csv)


if __name__ == '__main__':
    main()
