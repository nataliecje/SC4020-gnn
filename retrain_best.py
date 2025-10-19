"""Retrain a model with specific hyperparameters and save the best weights.

Usage example:
    python3 scripts/retrain_best.py --model GCN --num_layers 3 --hidden_channels 64 --dropout 0.2 --lr 0.005 --epochs 200

The script uses the same code in `simple_pipeline.py` (SimpleDataLoader, SimpleFeatureExtractor, model classes and SimpleTrainer).
"""
import argparse
import os
import torch

from simple_pipeline import SimpleDataLoader, SimpleFeatureExtractor

# Import model classes only if available
try:
    from simple_pipeline import SimpleGCN, SimpleGAT, SimpleGraphSAGE, SimpleGraphTransformer, SimpleTrainer
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False


def build_model(model_name, in_channels, out_channels, hidden_channels, num_layers, heads=4, dropout=0.5):
    if model_name == 'GCN':
        return SimpleGCN(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
    elif model_name == 'GAT':
        return SimpleGAT(in_channels, hidden_channels, out_channels, heads=heads, num_layers=num_layers, dropout=dropout)
    elif model_name == 'GraphSAGE':
        return SimpleGraphSAGE(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
    elif model_name == 'GraphTransformer':
        return SimpleGraphTransformer(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['GCN','GAT','GraphSAGE','GraphTransformer'])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--heads', type=int, default=4, help='GAT heads')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--from-csv', type=str, default=None, help="Path to hyper_search CSV or 'latest' to auto-find the newest hyper_search CSV in results/")
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    if not TORCH_OK:
        raise RuntimeError('PyTorch or model classes not available in simple_pipeline.py')

    os.makedirs('results', exist_ok=True)

    # Optionally override args from a hyper_search CSV
    if args.from_csv is not None:
        import glob, csv

        # determine csv path
        if args.from_csv == 'latest':
            files = glob.glob(os.path.join('results', 'hyper_search_*.csv'))
            if not files:
                raise FileNotFoundError("No hyper_search CSV files found in results/")
            csv_path = max(files, key=os.path.getmtime)
        else:
            csv_path = args.from_csv

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # read csv and find best row for this model
        best_row = None
        best_score = -float('inf')
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('model') != args.model:
                    continue
                try:
                    score = float(row.get('micro_f1', row.get('accuracy', 0)))
                except Exception:
                    score = float(row.get('accuracy', 0) or 0)
                if score > best_score:
                    best_score = score
                    best_row = row

        if best_row is None:
            raise ValueError(f"No rows for model {args.model} found in {csv_path}")

        # override args where possible
        def _maybe_set(name, cast):
            if name in best_row and best_row[name] not in (None, ''):
                try:
                    return cast(best_row[name])
                except Exception:
                    return getattr(args, name)
            return getattr(args, name)

        args.num_layers = _maybe_set('num_layers', int)
        args.hidden_channels = _maybe_set('hidden_channels', int)
        args.dropout = _maybe_set('dropout', float)
        args.lr = _maybe_set('lr', float)
        if args.model == 'GAT':
            args.heads = _maybe_set('heads', int)

        print(f"Loaded best config from {csv_path} for model {args.model}: num_layers={args.num_layers}, hidden_channels={args.hidden_channels}, dropout={args.dropout}, lr={args.lr}, heads={args.heads}")

    # Load data
    loader = SimpleDataLoader()
    edge_index, y, num_nodes, labels = loader.load_data()

    # Features
    feat_ex = SimpleFeatureExtractor(edge_index, num_nodes, y)
    x = feat_ex.get_structural_features()

    num_classes = int(y.max()) + 1

    # Build model and data wrapper expected by SimpleTrainer
    model = build_model(args.model, in_channels=x.size(1) if hasattr(x, 'size') else x.shape[1], out_channels=num_classes, hidden_channels=args.hidden_channels, num_layers=args.num_layers, heads=args.heads, dropout=args.dropout)

    # Create a lightweight Data object similar to simple_pipeline.Data
    try:
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
    except Exception:
        # Minimal fallback if torch_geometric unavailable
        class DataObj:
            pass
        data = DataObj()
        data.x = x
        data.edge_index = edge_index
        data.y = y
        data.num_nodes = num_nodes

    # Create train/val/test splits (reuse code from pipeline)
    labeled_nodes = (y >= 0).nonzero(as_tuple=False).view(-1).tolist()
    import random
    random.shuffle(labeled_nodes)
    n = len(labeled_nodes)
    train_idx = labeled_nodes[:int(0.6 * n)]
    val_idx = labeled_nodes[int(0.6 * n):int(0.8 * n)]
    test_idx = labeled_nodes[int(0.8 * n):]

    import torch
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    trainer = SimpleTrainer(model, data, lr=args.lr)

    print(f"Training {args.model} with layers={args.num_layers}, hidden={args.hidden_channels}, dropout={args.dropout}, lr={args.lr}, heads={args.heads}")
    train_losses, val_f1_scores = trainer.train(epochs=args.epochs, verbose=True)

    # After training, trainer.model should have best weights loaded (trainer.train loads best state if present). Save state dict.
    save_path = args.save_path or f'results/{args.model}_best_layers{args.num_layers}_hid{args.hidden_channels}_drop{args.dropout}_lr{args.lr}.pth'
    torch.save(trainer.model.state_dict(), save_path)

    # Evaluate on test
    test_micro_f1, test_macro_f1, test_acc, y_true, y_pred = trainer.evaluate(data.test_mask)

    print('Saved model to', save_path)
    print(f'Test micro-F1: {test_micro_f1:.4f}, Test accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    main()
