import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Try to import PyTorch components with fallbacks
try:
    import torch
    import torch.nn.functional as F
    from torch.nn import Dropout
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected, degree
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch/PyTorch Geometric not found. Please install dependencies.")
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not found. Please install dependencies.")
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    print("Warning: NetworkX not found. Some features may be limited.")
    NETWORKX_AVAILABLE = False

# Try to import the real Mamba from mamba-ssm
try:
    from mamba_ssm import Mamba
    print("Using real Mamba from mamba-ssm")
except Exception as e:
    print("mamba-ssm not found or failed to load. Using SimpleMambaBlock (CPU fallback).")
    
    class Mamba(torch.nn.Module):
        """
        Simple CPU fallback for Mamba block.
        Mimics sequence processing with lightweight linear layers.
        """
        def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
            super().__init__()
            self.fc1 = torch.nn.Linear(d_model, d_model)
            self.fc2 = torch.nn.Linear(d_model, d_model)
            self.dropout = torch.nn.Dropout(0.1)

        def forward(self, x):
            # x: (batch, seq_len, d_model)
            out = F.relu(self.fc1(x))
            out = self.dropout(out)
            out = self.fc2(out)
            return out


# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

if TORCH_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

class DataAnalyzer:
    """Analyze the dataset to understand its properties"""
    
    def __init__(self, edge_file="email-Eu-core.txt", label_file="email-Eu-core-department-labels.txt"):
        self.edge_file = edge_file
        self.label_file = label_file
        self.load_and_analyze()
    
    def load_and_analyze(self):
        print("Loading and analyzing dataset...")
        
        # Load edges
        edges = []
        with open(self.edge_file) as f:
            for line in f:
                a, b = map(int, line.split())
                edges.append((a, b))
        
        # Load labels
        labels = {}
        with open(self.label_file) as f:
            for line in f:
                n, lab = map(int, line.split())
                labels[n] = lab
        
        # Basic statistics
        all_nodes = set()
        for a, b in edges:
            all_nodes.add(a)
            all_nodes.add(b)
        
        labeled_nodes = set(labels.keys())
        
        print(f"Total nodes: {len(all_nodes)}")
        print(f"Total edges: {len(edges)}")
        print(f"Labeled nodes: {len(labeled_nodes)}")
        print(f"Total classes: {len(set(labels.values()))}")
        
        # Class distribution
        class_counts = defaultdict(int)
        for label in labels.values():
            class_counts[label] += 1
        
        print("\nClass distribution:")
        for class_id, count in sorted(class_counts.items()):
            print(f"  Class {class_id}: {count} nodes")
        
        self.edges = edges
        self.labels = labels
        self.all_nodes = all_nodes
        self.labeled_nodes = labeled_nodes
        self.class_counts = dict(class_counts)
        self.num_classes = len(set(labels.values()))

class SimpleDataLoader:
    """Simple data loader without heavy dependencies"""
    
    def __init__(self, edge_file="email-Eu-core.txt", label_file="email-Eu-core-department-labels.txt"):
        self.edge_file = edge_file
        self.label_file = label_file
        
    def load_data(self):
        # Load edges
        edges, labels, max_node = [], {}, -1
        with open(self.edge_file) as f:
            for line in f:
                a, b = map(int, line.split())
                edges.append([a, b])
                max_node = max(max_node, a, b)
        
        # Load labels
        with open(self.label_file) as f:
            for line in f:
                n, lab = map(int, line.split())
                labels[n] = lab
                max_node = max(max_node, n)
        
        num_nodes = max_node + 1
        
        if TORCH_AVAILABLE:
            edge_index = torch.tensor(edges).t().contiguous()
            edge_index = to_undirected(edge_index)
            
            # Create labels tensor
            y = torch.full((num_nodes,), -1, dtype=torch.long)
            for n, lab in labels.items():
                y[n] = lab
            
            return edge_index, y, num_nodes, labels
        else:
            return edges, labels, num_nodes, labels

class SimpleFeatureExtractor:
    """Simple feature extraction"""
    
    def __init__(self, edge_index, num_nodes, y):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.y = y
        
    def get_structural_features(self):
        """Get degree-based structural features"""
        if TORCH_AVAILABLE:
            deg = degree(self.edge_index[0], num_nodes=self.num_nodes).float()
            x = torch.stack([deg, torch.log1p(deg)], dim=1)
            x = (x - x.mean(0)) / (x.std(0) + 1e-9)
            return x
        else:
            # Fallback implementation
            degree_dict = defaultdict(int)
            for edge in self.edge_index:
                degree_dict[edge[0]] += 1
                degree_dict[edge[1]] += 1
            
            degrees = np.array([degree_dict[i] for i in range(self.num_nodes)], dtype=np.float32)
            log_degrees = np.log1p(degrees)
            features = np.column_stack([degrees, log_degrees])
            
            # Normalize
            features = (features - features.mean(0)) / (features.std(0) + 1e-9)
            return features
    
    def get_random_features(self, dim=64):
        """Generate random features as baseline"""
        if TORCH_AVAILABLE:
            return torch.randn(self.num_nodes, dim)
        else:
            return np.random.randn(self.num_nodes, dim)

# Define models only if PyTorch is available
if TORCH_AVAILABLE:
    class SimpleGCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
            super().__init__()
            self.num_layers = num_layers
            self.dropout = dropout
            
            self.convs = torch.nn.ModuleList()
            self.convs.append(GCNConv(in_channels, hidden_channels))
            
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
            if num_layers == 1:
                self.convs = torch.nn.ModuleList([GCNConv(in_channels, out_channels)])
            else:
                self.convs.append(GCNConv(hidden_channels, out_channels))
            
        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.convs[-1](x, edge_index)
            return x

    class SimpleGAT(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, heads=4, num_layers=2, dropout=0.5):
            super().__init__()
            self.num_layers = num_layers
            self.dropout = dropout
            
            self.convs = torch.nn.ModuleList()
            
            if num_layers == 1:
                self.convs.append(GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout))
            else:
                self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
                
                for _ in range(num_layers - 2):
                    self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
                
                self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
            
        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.convs[-1](x, edge_index)
            return x
        
    class SimpleGraphSAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels,
                     num_layers=2, dropout=0.5, aggregator='mean'):
            super().__init__()
            self.num_layers = num_layers
            self.dropout = dropout
            self.aggregator = aggregator  # 'mean', 'max', 'pool', or 'lstm'

            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
            
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
            
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregator))
        
        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)
            return x
    
    # HYBRID MODEL
    class SimpleGraphTransformer(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels,
                    num_layers=2, dropout=0.3,
                    nhead=2, num_transformer_layers=1,
                    dim_feedforward=None):
            """
            Optimized Graph Transformer (lightweight, CPU/GPU compatible)
            - fewer attention heads
            - smaller feedforward layers
            - less dropout
            - efficient residual structure
            """
            super().__init__()
            self.dropout = dropout

            # Local GCN layers
            self.local_convs = torch.nn.ModuleList()
            self.local_convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.local_convs.append(GCNConv(hidden_channels, hidden_channels))

            # Lightweight Transformer Encoder
            if dim_feedforward is None:
                dim_feedforward = hidden_channels * 2  # smaller feedforward size

            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_channels,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu"
            )
            self.global_block = TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

            # Output classifier
            self.fc_out = torch.nn.Linear(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            # Local message passing
            for conv in self.local_convs:
                x = conv(x, edge_index)
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            # Treat nodes as sequence for Transformer
            seq = x.unsqueeze(0)  # (1, N, hidden)
            global_ctx = self.global_block(seq)  # (1, N, hidden)

            # Residual fusion
            x = x + global_ctx.squeeze(0)
            return self.fc_out(x)

class SimpleTrainer:
    """Simple trainer with minimal dependencies"""
    
    def __init__(self, model, data, lr=0.01, weight_decay=5e-4):
        if TORCH_AVAILABLE:
            self.model = model.to(DEVICE)
            self.data = data.to(DEVICE)
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise RuntimeError("PyTorch not available")
        
        self.train_losses = []
        self.val_f1_scores = []
        
    def train_epoch(self, num_batches=2):
        self.model.train()
        train_nodes = self.data.train_mask.nonzero(as_tuple=False).view(-1)

        # Initialize fixed node splits (only once)
        if not hasattr(self, 'fixed_batches'):
            perm = torch.randperm(len(train_nodes))
            split_size = len(train_nodes) // num_batches
            self.fixed_batches = [
                train_nodes[perm[i*split_size:(i+1)*split_size]] 
                for i in range(num_batches - 1)
            ]
            # last batch gets any leftovers
            self.fixed_batches.append(train_nodes[perm[(num_batches-1)*split_size:]])

        batch_losses = []
        for batch_idx in self.fixed_batches:
            self.optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out[batch_idx], self.data.y[batch_idx])
            loss.backward()
            self.optimizer.step()
            batch_losses.append(loss.item())

        # Return averaged loss for smoother tracking
        return np.mean(batch_losses)
        
    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1)
            
            y_true = self.data.y[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()
            
            if SKLEARN_AVAILABLE:
                micro_f1 = f1_score(y_true, y_pred, average='micro')
                macro_f1 = f1_score(y_true, y_pred, average='macro')
                accuracy = accuracy_score(y_true, y_pred)
            else:
                # Simple accuracy calculation
                accuracy = np.mean(y_true == y_pred)
                micro_f1 = accuracy  # Approximation
                macro_f1 = accuracy  # Approximation
            
            return micro_f1, macro_f1, accuracy, y_true, y_pred
    
    def train(self, epochs=200, patience=50, verbose=True):
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            loss = self.train_epoch()
            self.train_losses.append(loss)
            
            # Evaluate on validation set
            val_micro_f1, val_macro_f1, val_acc, _, _ = self.evaluate(self.data.val_mask)
            self.val_f1_scores.append(val_micro_f1)
            
            if val_micro_f1 > best_val_f1:
                best_val_f1 = val_micro_f1
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val F1 {val_micro_f1:.3f}")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.train_losses, self.val_f1_scores

class ExperimentPipeline:
    """Main experiment pipeline"""
    
    def __init__(self):
        self.results = {}
        
        # Load and analyze data
        self.analyzer = DataAnalyzer()
        
        if TORCH_AVAILABLE:
            # Load data for PyTorch
            data_loader = SimpleDataLoader()
            self.edge_index, self.y, self.num_nodes, self.labels = data_loader.load_data()
            
            # Create train/val/test splits
            self.create_splits()
            
            # Initialize feature extractor
            self.feature_extractor = SimpleFeatureExtractor(self.edge_index, self.num_nodes, self.y)
        
    def create_splits(self):
        # Get nodes with labels
        labeled_nodes = (self.y >= 0).nonzero(as_tuple=False).view(-1).tolist()
        random.shuffle(labeled_nodes)
        
        n = len(labeled_nodes)
        train_idx = labeled_nodes[:int(0.6 * n)]
        val_idx = labeled_nodes[int(0.6 * n):int(0.8 * n)]
        test_idx = labeled_nodes[int(0.8 * n):]
        
        self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        
        self.train_mask[train_idx] = True
        self.val_mask[val_idx] = True
        self.test_mask[test_idx] = True
        
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
    def run_single_experiment(self, model_type, **kwargs):
        """Run a single experiment"""
        if not TORCH_AVAILABLE:
            print("Cannot run experiments without PyTorch")
            return None
            
        print(f"Running {model_type} experiment...")
        
        # Get structural features (simple baseline)
        x = self.feature_extractor.get_structural_features()
        
        # Create data object
        data = Data(x=x, edge_index=self.edge_index, y=self.y, num_nodes=self.num_nodes)
        data.train_mask = self.train_mask
        data.val_mask = self.val_mask
        data.test_mask = self.test_mask
        
        # Get number of classes
        num_classes = int(self.y.max()) + 1
        
        # Create model
        if model_type == 'GCN':
            model = SimpleGCN(
                in_channels=x.size(1),
                hidden_channels=kwargs.get('hidden_channels', 64),
                out_channels=num_classes,
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.5)
            )
        elif model_type == 'GAT':
            model = SimpleGAT(
                in_channels=x.size(1),
                hidden_channels=kwargs.get('hidden_channels', 32),
                out_channels=num_classes,
                heads=kwargs.get('heads', 4),
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.5)
            )
        elif model_type == 'GraphSAGE':
            model = SimpleGraphSAGE(
                in_channels=x.size(1),
                hidden_channels=kwargs.get('hidden_channels', 64),
                out_channels=num_classes,
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.5),
                aggregator=kwargs.get('aggregator', 'mean')
            )
        elif model_type == 'GraphTransformer': 
            model = SimpleGraphTransformer(
                in_channels=x.size(1),
                hidden_channels=kwargs.get('hidden_channels', 64),
                out_channels=num_classes,
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.3),
                nhead=kwargs.get('nhead', 2),
                num_transformer_layers=kwargs.get('num_transformer_layers', 1),
                dim_feedforward=kwargs.get('dim_feedforward', 128)
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        
        # Train model
        trainer = SimpleTrainer(model, data, lr=kwargs.get('lr', 0.01))
        train_losses, val_f1_scores = trainer.train(epochs=kwargs.get('epochs', 200), verbose=False)
        
        # Evaluate on test set
        test_micro_f1, test_macro_f1, test_acc, y_true, y_pred = trainer.evaluate(data.test_mask)
        
        return {
            'test_micro_f1': test_micro_f1,
            'test_macro_f1': test_macro_f1,
            'test_accuracy': test_acc,
            'train_losses': train_losses,
            'val_f1_scores': val_f1_scores,
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist()
        }
    
    def run_table_a_experiments(self):
        """Table A: Main comparison"""
        print("\n=== Running Table A Experiments ===")
        
        results = {}
        for model in ['GCN', 'GAT', 'GraphSAGE', 'GraphTransformer']:
            result = self.run_single_experiment(model)
            if result:
                results[model] = {
                    'micro_f1': result['test_micro_f1'],
                    'macro_f1': result['test_macro_f1'],
                    'accuracy': result['test_accuracy'],
                    'train_losses': result['train_losses'],
                    'val_f1_scores': result['val_f1_scores']
                }
        
        self.results['table_a'] = results
        return results
        
    def run_ablation_studies(self):
        """Run ablation studies"""
        print("\n=== Running Ablation Studies ===")
        
        results = {}

        # Depth ablation for all models
        print("Depth ablation...")
        results['depth'] = {}
        for model_type in ['GCN', 'GAT', 'GraphSAGE', 'GraphTransformer']:
            results['depth'][model_type] = {}
            for depth in [1, 2, 3]:
                result = self.run_single_experiment(model_type, num_layers=depth)
                if result:
                    results['depth'][model_type][depth] = {
                        'micro_f1': result['test_micro_f1'],
                        'macro_f1': result['test_macro_f1'],
                        'accuracy': result['test_accuracy']
                    }

        # Dropout ablation for all models
        print("Dropout ablation...")
        results['dropout'] = {}
        for model_type in ['GCN', 'GAT', 'GraphSAGE', 'GraphTransformer']:
            results['dropout'][model_type] = {}
            for dropout in [0.0, 0.3, 0.5, 0.7]:
                result = self.run_single_experiment(model_type, dropout=dropout)
                if result:
                    results['dropout'][model_type][dropout] = {
                        'micro_f1': result['test_micro_f1'],
                        'macro_f1': result['test_macro_f1'],
                        'accuracy': result['test_accuracy']
                    }

        # Heads ablation for GAT
        print("Heads ablation for GAT...")
        results['heads'] = {}
        for heads in [1, 2, 4, 8]:
            result = self.run_single_experiment('GAT', heads=heads)
            if result:
                results['heads'][heads] = {
                    'micro_f1': result['test_micro_f1'],
                    'macro_f1': result['test_macro_f1'],
                    'accuracy': result['test_accuracy']
                }
        self.results['ablations'] = results
        return results

    
    def create_visualizations(self):
        """Create visualizations"""
        print("\n=== Creating Visualizations ===")
        
        os.makedirs('results', exist_ok=True)
        
    def _smooth_curve(self, values, window=5):
        """Simple moving average smoothing that preserves original length."""
        try:
            vals = np.array(values)
        except Exception:
            return values
        if vals.size < window or window <= 1:
            return vals
        weights = np.ones(window) / window
        smoothed = np.convolve(vals, weights, mode='valid')
        pad = np.full(len(vals) - len(smoothed), vals[0])
        return np.concatenate([pad, smoothed])

        # Training curves
        if 'table_a' in self.results:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            for model, data in self.results['table_a'].items():
                if 'train_losses' in data:
                    plt.plot(data['train_losses'], label=f'{model}')
            plt.xlabel('Epoch')
            plt.ylabel('Training Loss')
            plt.title('Training Loss Curves')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            for model, data in self.results['table_a'].items():
                if 'val_f1_scores' in data:
                    raw = data['val_f1_scores']
                    sm = self._smooth_curve(raw, window=5)
                    plt.plot(sm, label=f'{model}')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Micro-F1')
            plt.title('Validation F1 Curves')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Ablation studies
        if 'ablations' in self.results:
            for study_name, study_data in self.results['ablations'].items():
                plt.figure(figsize=(8, 5))
                
                # Depth / Dropout for all models
                if study_name in ['depth', 'dropout']:
                    for model_type, model_results in study_data.items():
                        x_vals = list(model_results.keys())
                        y_vals = [model_results[k]['micro_f1'] for k in x_vals]
                        plt.plot(x_vals, y_vals, 'o-', label=model_type)
                    
                    xlabel = 'Number of Layers' if study_name == 'depth' else 'Dropout Rate'
                    plt.xlabel(xlabel)
                    plt.ylabel('Test Micro-F1')
                    plt.title(f'{study_name.title()} Ablation')
                    plt.grid(True)
                    plt.legend()
                
                # Heads ablation (GAT)
                elif study_name == 'heads':
                    x_vals = list(study_data.keys())
                    y_vals = [study_data[k]['micro_f1'] for k in x_vals]
                    plt.plot(x_vals, y_vals, 'o-')
                    plt.xlabel('Number of Heads')
                    plt.ylabel('Test Micro-F1')
                    plt.title('Heads Ablation (GAT)')
                    plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f'results/ablation_{study_name}.png', dpi=300, bbox_inches='tight')
                plt.close()

        # Early stopping continuation plots
        if 'table_a' in self.results:
            for model, data in self.results['table_a'].items():
                plt.figure(figsize=(6, 4))
                
                train_losses = data.get('train_losses', [])
                val_f1_scores = data.get('val_f1_scores', [])
                epochs = list(range(len(train_losses)))
                
                # Plot actual training
                if train_losses:
                    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
                if val_f1_scores:
                    sm_val = self._smooth_curve(val_f1_scores, window=5)
                    plt.plot(epochs, sm_val, label='Val Micro-F1', color='green')
                
                # Early stopping epoch
                early_stop_epoch = len(train_losses) - 1
                plt.axvline(x=early_stop_epoch, color='red', linestyle='--', label='Early Stop')
                
                # Hypothetical continuation (next 20 epochs)
                hypothetical_epochs = list(range(early_stop_epoch + 1, early_stop_epoch + 21))
                if train_losses:
                    last_loss = train_losses[-1]
                    continued_loss = [last_loss * (0.99 ** (i+1)) for i in range(len(hypothetical_epochs))]
                    plt.plot(hypothetical_epochs, continued_loss, 'b--', alpha=0.7, label='Hypothetical Train Loss')
                if val_f1_scores:
                    last_f1 = val_f1_scores[-1]
                    continued_f1 = [min(last_f1 + 0.001*(i+1), 1.0) for i in range(len(hypothetical_epochs))]
                    plt.plot(hypothetical_epochs, continued_f1, 'g--', alpha=0.7, label='Hypothetical Val F1')
                
                plt.title(f'{model} Training with Early Stopping and Continuation')
                plt.xlabel('Epoch')
                plt.ylabel('Metric')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'results/{model}_early_stopping_continuation.png', dpi=300, bbox_inches='tight')
                plt.close()

    
    def save_results(self):
        """Save results to files"""
        print("\n=== Saving Results ===")
        
        os.makedirs('results', exist_ok=True)
        
        # Save raw results as JSON
        with open('results/all_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary tables
        if 'table_a' in self.results:
            df = pd.DataFrame.from_dict(self.results['table_a'], orient='index')
            df.to_csv('results/table_a_main_results.csv')
            print("\nTable A - Main Results:")
            print(df[['micro_f1', 'macro_f1', 'accuracy']].round(4))
        
        if 'ablations' in self.results:
            for study_name, study_data in self.results['ablations'].items():
                df = pd.DataFrame.from_dict(study_data, orient='index')
                df.to_csv(f'results/ablation_{study_name}.csv')
                print(f"\nAblation Study - {study_name.title()}:")
                print(df.round(4))
    
    def run_all_experiments(self):
        """Run all experiments"""
        print("Starting GNN Evaluation Pipeline")
        print("=" * 50)
        
        if not TORCH_AVAILABLE:
            print("ERROR: PyTorch not available. Please install required dependencies.")
            print("Run: pip install torch torch-geometric scikit-learn")
            return
        
        # Run experiments
        self.run_table_a_experiments()
        self.run_ablation_studies()
        
        # Create visualizations and save results
        self.create_visualizations()
        self.save_results()
        
        print("\n" + "=" * 50)
        print("All experiments completed!")
        print("Results saved in 'results/' directory")
        
        return self.results

if __name__ == "__main__":
    pipeline = ExperimentPipeline()
    results = pipeline.run_all_experiments()
