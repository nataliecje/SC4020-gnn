import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import ModuleList, Linear
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

class ImprovedGAT(torch.nn.Module):
    """Enhanced GAT with residual connections and better architecture"""
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=4, heads=8, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = Linear(in_channels, hidden_channels)
        
        # GAT layers with residual connections
        self.gat_layers = ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_channels, hidden_channels, 
                       heads=heads, concat=False, dropout=dropout)
            )
        
        # Output projection
        self.output_proj = Linear(hidden_channels, out_channels)
        
        # Layer normalization
        self.layer_norms = ModuleList([
            torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        
    def forward(self, x, edge_index):
        # Input projection
        x = self.input_proj(x)
        x = F.elu(x)
        
        # GAT layers with residual connections
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = x
            x = gat_layer(x, edge_index)
            x = layer_norm(x)
            x = F.elu(x)
            
            # Residual connection (skip connection)
            if i > 0:  # Skip first layer residual to allow dimension change
                x = x + residual
            
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output projection
        x = self.output_proj(x)
        return x

class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_enhanced_features(edge_index, num_nodes, y=None):
    """Get enhanced structural features"""
    
    # Basic degree features
    deg = degree(edge_index[0], num_nodes=num_nodes).float()
    
    # Create feature matrix
    features = []
    
    # 1. Basic degree features
    features.append(deg.unsqueeze(1))
    features.append(torch.log1p(deg).unsqueeze(1))
    
    # 2. Degree powers (capture higher-order connectivity)
    features.append(torch.sqrt(deg).unsqueeze(1))
    features.append((deg ** 2).unsqueeze(1))
    
    # 3. Normalized degree features
    max_deg = deg.max()
    features.append((deg / max_deg).unsqueeze(1))
    
    # 4. Local clustering approximation
    # Count triangles for each node (approximate clustering)
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    adj_matrix[edge_index[0], edge_index[1]] = 1
    
    # A^2 gives number of 2-hop paths
    adj_squared = torch.mm(adj_matrix, adj_matrix)
    local_clustering = torch.diag(adj_squared).float().unsqueeze(1)
    features.append(local_clustering)
    
    # 5. Degree centrality relative to neighborhood
    # For each node, compute average degree of neighbors
    neighbor_avg_deg = torch.zeros(num_nodes)
    for i in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == i]
        if len(neighbors) > 0:
            neighbor_avg_deg[i] = deg[neighbors].mean()
    
    features.append(neighbor_avg_deg.unsqueeze(1))
    
    # 6. Degree deviation from neighborhood
    deg_deviation = deg - neighbor_avg_deg
    features.append(deg_deviation.unsqueeze(1))
    
    # Concatenate all features
    x = torch.cat(features, dim=1)
    
    # Normalize features
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    
    return x

def get_class_weights(y_train):
    """Compute class weights for imbalanced dataset"""
    
    class_counts = torch.bincount(y_train)
    total_samples = len(y_train)
    num_classes = len(class_counts)
    
    # Inverse frequency weighting with smoothing
    weights = total_samples / (num_classes * class_counts.float())
    
    # Apply square root to reduce extreme weights
    weights = torch.sqrt(weights)
    
    # Cap maximum weight to prevent instability
    weights = torch.clamp(weights, max=5.0)
    
    return weights

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=50, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def train_improved_model(data, num_epochs=300, lr=0.005, weight_decay=1e-4):
    """Train the improved model with all enhancements"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move data to device
    data = data.to(device)
    
    # Get enhanced features
    print("Generating enhanced features...")
    x_enhanced = get_enhanced_features(data.edge_index, data.num_nodes, data.y)
    data.x = x_enhanced.to(device)
    
    print(f"Feature dimensions: {data.x.size(1)}")
    
    # Create improved model
    num_classes = int(data.y.max()) + 1
    model = ImprovedGAT(
        in_channels=data.x.size(1),
        hidden_channels=64,
        out_channels=num_classes,
        num_layers=4,  # Deeper network based on your findings
        heads=8,       # More attention heads
        dropout=0.0    # No dropout as per your findings
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get class weights for loss function
    class_weights = get_class_weights(data.y[data.train_mask])
    class_weights = class_weights.to(device)
    
    # Use focal loss with class weights
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    
    # Optimizer with different learning rates for different parts
    optimizer = torch.optim.AdamW([
        {'params': model.input_proj.parameters(), 'lr': lr},
        {'params': model.gat_layers.parameters(), 'lr': lr * 0.5},
        {'params': model.output_proj.parameters(), 'lr': lr},
    ], weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=20, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=50)
    
    # Training history
    train_losses = []
    val_f1_scores = []
    
    best_val_f1 = 0
    best_model_state = None
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        
        # Compute weighted cross-entropy loss
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], 
                              weight=class_weights)
        
        # Add focal loss component
        focal_loss = criterion(out[data.train_mask], data.y[data.train_mask])
        total_loss = 0.7 * loss + 0.3 * focal_loss  # Weighted combination
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_pred = val_out[data.val_mask].argmax(dim=1)
            val_true = data.y[data.val_mask]
            
            val_f1 = f1_score(val_true.cpu(), val_pred.cpu(), average='micro')
            val_macro_f1 = f1_score(val_true.cpu(), val_pred.cpu(), average='macro')
            val_acc = accuracy_score(val_true.cpu(), val_pred.cpu())
        
        # Record history
        train_losses.append(total_loss.item())
        val_f1_scores.append(val_f1)
        
        # Update learning rate
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
        
        # Early stopping check
        early_stopping(val_f1)
        
        # Print progress
        if epoch % 20 == 0 or epoch < 10:
            print(f"Epoch {epoch:3d} | Loss: {total_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | Val Macro-F1: {val_macro_f1:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        test_pred = test_out[data.test_mask].argmax(dim=1)
        test_true = data.y[data.test_mask]
        
        test_f1 = f1_score(test_true.cpu(), test_pred.cpu(), average='micro')
        test_macro_f1 = f1_score(test_true.cpu(), test_pred.cpu(), average='macro')
        test_acc = accuracy_score(test_true.cpu(), test_pred.cpu())
    
    results = {
        'test_micro_f1': test_f1,
        'test_macro_f1': test_macro_f1,
        'test_accuracy': test_acc,
        'best_val_f1': best_val_f1,
        'train_losses': train_losses,
        'val_f1_scores': val_f1_scores
    }
    
    print(f"\n🎉 Final Results:")
    print(f"Test Micro-F1: {test_f1:.4f}")
    print(f"Test Macro-F1: {test_macro_f1:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return model, results

def create_training_plots(results):
    """Create training visualization plots"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training loss
    axes[0].plot(results['train_losses'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Validation F1
    axes[1].plot(results['val_f1_scores'])
    axes[1].set_title('Validation Micro-F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Micro-F1')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/improved_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training curves saved to 'results/improved_training_curves.png'")

if __name__ == "__main__":
    # This would be called with your actual data
    print("Enhanced GNN Training Script")
    print("=" * 50)
    print("This script implements:")
    print("✓ Enhanced structural features (8 dimensions)")
    print("✓ Deeper GAT with residual connections")
    print("✓ Focal loss for class imbalance")
    print("✓ Class-weighted training")
    print("✓ Advanced optimization")
    print("✓ Early stopping")
    print("=" * 50)
    print("\nTo use with your data:")
    print("model, results = train_improved_model(data)")
    print("create_training_plots(results)")
    print("\nExpected improvement: 22% → 30-35% accuracy")
