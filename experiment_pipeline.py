import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.transforms import LaplacianLambdaMax
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from node2vec import Node2Vec
import networkx as nx
from scipy.sparse import coo_matrix
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class DataLoader:
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
        edge_index = torch.tensor(edges).t().contiguous()
        edge_index = to_undirected(edge_index)
        
        # Create labels tensor
        y = torch.full((num_nodes,), -1, dtype=torch.long)
        for n, lab in labels.items():
            y[n] = lab
        
        return edge_index, y, num_nodes, labels

class FeatureExtractor:
    def __init__(self, edge_index, num_nodes, y):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.y = y
        
    def get_structural_features(self):
        """Get degree-based structural features"""
        deg = degree(self.edge_index[0], num_nodes=self.num_nodes).float()
        x = torch.stack([deg, torch.log1p(deg)], dim=1)
        x = (x - x.mean(0)) / (x.std(0) + 1e-9)
        return x
    
    def get_laplacian_pe(self, k=8):
        """Get Laplacian Positional Encoding"""
        try:
            from torch_geometric.utils import get_laplacian
            from torch.sparse import mm
            
            # Get normalized Laplacian
            edge_index, edge_weight = get_laplacian(self.edge_index, normalization='sym', num_nodes=self.num_nodes)
            
            # Convert to dense for eigendecomposition
            L = torch.sparse_coo_tensor(edge_index, edge_weight, (self.num_nodes, self.num_nodes)).to_dense()
            
            # Compute eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(L)
            
            # Take the k smallest non-zero eigenvalues/eigenvectors
            idx = eigenvals.argsort()[1:k+1]  # Skip the first (zero) eigenvalue
            pe = eigenvecs[:, idx]
            
            return pe
        except:
            # Fallback to random features if Laplacian PE fails
            print("Warning: Laplacian PE failed, using random features")
            return torch.randn(self.num_nodes, k)
    
    def get_node2vec_features(self, dimensions=64, walk_length=30, num_walks=200, workers=1):
        """Get Node2Vec embeddings"""
        try:
            # Convert to NetworkX graph
            edges_np = self.edge_index.numpy()
            G = nx.Graph()
            G.add_edges_from(edges_np.T)
            
            # Generate Node2Vec embeddings
            node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, 
                              num_walks=num_walks, workers=workers, quiet=True)
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            
            # Extract embeddings for all nodes
            embeddings = torch.zeros(self.num_nodes, dimensions)
            for i in range(self.num_nodes):
                if i in model.wv:
                    embeddings[i] = torch.tensor(model.wv[i])
                else:
                    embeddings[i] = torch.randn(dimensions) * 0.1  # Random for missing nodes
            
            return embeddings
        except Exception as e:
            print(f"Warning: Node2Vec failed ({e}), using random features")
            return torch.randn(self.num_nodes, dimensions)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
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

class Trainer:
    def __init__(self, model, data, lr=0.01, weight_decay=5e-4):
        self.model = model.to(DEVICE)
        self.data = data.to(DEVICE)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.train_losses = []
        self.val_f1_scores = []
        
    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1)
            
            y_true = self.data.y[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()
            
            micro_f1 = f1_score(y_true, y_pred, average='micro')
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)
            
            return micro_f1, macro_f1, accuracy, y_true, y_pred
    
    def train(self, epochs=200, patience=50):
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            loss = self.train_epoch()
            self.train_losses.append(loss)
            
            # Evaluate on validation set
            val_micro_f1, val_macro_f1, val_acc, _, _ = self.evaluate(self.data.val_mask)
            self.val_f1_scores.append(val_micro_f1)
            
            if val_micro_f1 > best_val_f1:
                best_val_f1 = val_micro_f1
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        return self.train_losses, self.val_f1_scores

class ExperimentRunner:
    def __init__(self):
        self.results = {
            'table_a': [],
            'table_b': [],
            'table_c': [],
            'training_curves': {}
        }
        
        # Load data
        data_loader = DataLoader()
        self.edge_index, self.y, self.num_nodes, self.labels = data_loader.load_data()
        
        # Create train/val/test splits
        self.create_splits()
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.edge_index, self.num_nodes, self.y)
        
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
        
    def run_experiment(self, model_type, feature_type, **kwargs):
        print(f"Running {model_type} with {feature_type} features...")
        
        # Get features
        if feature_type == 'structural':
            x = self.feature_extractor.get_structural_features()
        elif feature_type == 'laplacian_pe':
            x = self.feature_extractor.get_laplacian_pe()
        elif feature_type == 'node2vec':
            x = self.feature_extractor.get_node2vec_features()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Create data object
        data = Data(x=x, edge_index=self.edge_index, y=self.y, num_nodes=self.num_nodes)
        data.train_mask = self.train_mask
        data.val_mask = self.val_mask
        data.test_mask = self.test_mask
        
        # Get number of classes
        num_classes = int(self.y.max()) + 1
        
        # Create model
        if model_type == 'GCN':
            model = GCN(
                in_channels=x.size(1),
                hidden_channels=kwargs.get('hidden_channels', 64),
                out_channels=num_classes,
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.5)
            )
        elif model_type == 'GAT':
            model = GAT(
                in_channels=x.size(1),
                hidden_channels=kwargs.get('hidden_channels', 32),
                out_channels=num_classes,
                heads=kwargs.get('heads', 4),
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.5)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        trainer = Trainer(model, data, lr=kwargs.get('lr', 0.01))
        train_losses, val_f1_scores = trainer.train(epochs=kwargs.get('epochs', 200))
        
        # Evaluate on test set
        test_micro_f1, test_macro_f1, test_acc, y_true, y_pred = trainer.evaluate(data.test_mask)
        
        return {
            'test_micro_f1': test_micro_f1,
            'test_macro_f1': test_macro_f1,
            'test_accuracy': test_acc,
            'train_losses': train_losses,
            'val_f1_scores': val_f1_scores,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def run_table_a_experiments(self):
        """Table A: Main results comparing GCN vs GAT with best features"""
        print("Running Table A experiments...")
        
        models = ['GCN', 'GAT']
        features = ['structural', 'laplacian_pe', 'node2vec']
        
        best_results = {}
        
        for model in models:
            best_f1 = 0
            best_result = None
            best_feature = None
            
            for feature in features:
                result = self.run_experiment(model, feature)
                
                if result['test_micro_f1'] > best_f1:
                    best_f1 = result['test_micro_f1']
                    best_result = result
                    best_feature = feature
            
            best_results[model] = {
                'feature': best_feature,
                'micro_f1': best_result['test_micro_f1'],
                'macro_f1': best_result['test_macro_f1'],
                'accuracy': best_result['test_accuracy'],
                'train_losses': best_result['train_losses'],
                'val_f1_scores': best_result['val_f1_scores']
            }
        
        self.results['table_a'] = best_results
        self.results['training_curves']['table_a'] = best_results
        
    def run_table_b_experiments(self):
        """Table B: Feature comparison for GCN"""
        print("Running Table B experiments...")
        
        features = ['structural', 'laplacian_pe', 'node2vec']
        results = {}
        
        for feature in features:
            result = self.run_experiment('GCN', feature, hidden_channels=64)
            results[feature] = {
                'micro_f1': result['test_micro_f1'],
                'macro_f1': result['test_macro_f1'],
                'accuracy': result['test_accuracy']
            }
        
        self.results['table_b'] = results
        
    def run_table_c_experiments(self):
        """Table C: Ablation studies"""
        print("Running Table C experiments...")
        
        results = {}
        
        # (i) Depth ablation for GCN
        depths = [1, 2, 3, 4]
        results['depth'] = {}
        for depth in depths:
            result = self.run_experiment('GCN', 'structural', num_layers=depth)
            results['depth'][depth] = {
                'micro_f1': result['test_micro_f1'],
                'macro_f1': result['test_macro_f1'],
                'accuracy': result['test_accuracy']
            }
        
        # (ii) Dropout ablation for GCN
        dropouts = [0.0, 0.2, 0.5, 0.8]
        results['dropout'] = {}
        for dropout in dropouts:
            result = self.run_experiment('GCN', 'structural', dropout=dropout)
            results['dropout'][dropout] = {
                'micro_f1': result['test_micro_f1'],
                'macro_f1': result['test_macro_f1'],
                'accuracy': result['test_accuracy']
            }
        
        # (iii) Heads ablation for GAT
        heads = [1, 2, 4, 8]
        results['heads'] = {}
        for head in heads:
            result = self.run_experiment('GAT', 'structural', heads=head)
            results['heads'][head] = {
                'micro_f1': result['test_micro_f1'],
                'macro_f1': result['test_macro_f1'],
                'accuracy': result['test_accuracy']
            }
        
        self.results['table_c'] = results
        
    def generate_per_class_f1(self):
        """Generate per-class F1 scores for Figure 3"""
        print("Generating per-class F1 scores...")
        
        # Run both models with best features
        gcn_result = self.run_experiment('GCN', 'node2vec')  # Assuming node2vec is best
        gat_result = self.run_experiment('GAT', 'node2vec')
        
        # Calculate per-class F1 scores
        gcn_f1_per_class = f1_score(gcn_result['y_true'], gcn_result['y_pred'], average=None)
        gat_f1_per_class = f1_score(gat_result['y_true'], gat_result['y_pred'], average=None)
        
        # Get class support (number of samples per class)
        unique_classes, class_counts = np.unique(gcn_result['y_true'], return_counts=True)
        
        per_class_results = pd.DataFrame({
            'class': unique_classes,
            'support': class_counts,
            'gcn_f1': gcn_f1_per_class,
            'gat_f1': gat_f1_per_class
        })
        
        # Sort by support and take top-k classes
        per_class_results = per_class_results.sort_values('support', ascending=False)
        
        self.results['per_class_f1'] = per_class_results
        
    def create_visualizations(self):
        """Create all figures"""
        print("Creating visualizations...")
        
        # Create output directory
        os.makedirs('results', exist_ok=True)
        
        # Figure 1: Training loss curves
        plt.figure(figsize=(10, 6))
        if 'table_a' in self.results['training_curves']:
            for model, data in self.results['training_curves']['table_a'].items():
                plt.plot(data['train_losses'], label=f'{model} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves: GCN vs GAT')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/figure1_training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Validation micro-F1 curves
        plt.figure(figsize=(10, 6))
        if 'table_a' in self.results['training_curves']:
            for model, data in self.results['training_curves']['table_a'].items():
                plt.plot(data['val_f1_scores'], label=f'{model} Validation Micro-F1')
        plt.xlabel('Epoch')
        plt.ylabel('Micro-F1')
        plt.title('Validation Micro-F1 Curves: GCN vs GAT')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/figure2_validation_f1.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Per-class F1 scores
        if 'per_class_f1' in self.results:
            top_k = min(10, len(self.results['per_class_f1']))
            top_classes = self.results['per_class_f1'].head(top_k)
            
            plt.figure(figsize=(12, 8))
            x = np.arange(len(top_classes))
            width = 0.35
            
            plt.bar(x - width/2, top_classes['gcn_f1'], width, label='GCN', alpha=0.8)
            plt.bar(x + width/2, top_classes['gat_f1'], width, label='GAT', alpha=0.8)
            
            plt.xlabel('Class')
            plt.ylabel('F1 Score')
            plt.title(f'Per-Class F1 Scores (Top-{top_k} Classes by Support)')
            plt.xticks(x, [f'Class {c}\n(n={s})' for c, s in zip(top_classes['class'], top_classes['support'])])
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('results/figure3_per_class_f1.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_tables(self):
        """Save all tables to CSV files"""
        print("Saving tables...")
        
        os.makedirs('results', exist_ok=True)
        
        # Table A
        if 'table_a' in self.results:
            table_a_df = pd.DataFrame.from_dict(self.results['table_a'], orient='index')
            table_a_df.to_csv('results/table_a_main_results.csv')
            print("\nTable A - Main Results:")
            print(table_a_df[['feature', 'micro_f1', 'macro_f1', 'accuracy']])
        
        # Table B
        if 'table_b' in self.results:
            table_b_df = pd.DataFrame.from_dict(self.results['table_b'], orient='index')
            table_b_df.to_csv('results/table_b_feature_comparison.csv')
            print("\nTable B - Feature Comparison (GCN):")
            print(table_b_df)
        
        # Table C
        if 'table_c' in self.results:
            for ablation_type, data in self.results['table_c'].items():
                df = pd.DataFrame.from_dict(data, orient='index')
                df.to_csv(f'results/table_c_{ablation_type}_ablation.csv')
                print(f"\nTable C - {ablation_type.title()} Ablation:")
                print(df)
        
        # Per-class F1
        if 'per_class_f1' in self.results:
            self.results['per_class_f1'].to_csv('results/per_class_f1_scores.csv', index=False)
    
    def run_all_experiments(self):
        """Run all experiments"""
        print("Starting comprehensive GNN evaluation pipeline...")
        
        self.run_table_a_experiments()
        self.run_table_b_experiments()
        self.run_table_c_experiments()
        self.generate_per_class_f1()
        
        self.create_visualizations()
        self.save_tables()
        
        print("\nAll experiments completed! Results saved in 'results/' directory.")
        return self.results

if __name__ == "__main__":
    # Run all experiments
    runner = ExperimentRunner()
    results = runner.run_all_experiments()
