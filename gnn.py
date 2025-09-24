import os, random, math
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
from torch_geometric.nn import MessagePassing

# Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EDGE_FILE = "email-Eu-core.txt"
LABEL_FILE = "email-Eu-core-department-labels.txt"

# Load dataset
edges = []
max_node = -1
with open(EDGE_FILE) as f:
    for line in f:
        a, b = map(int, line.split())
        edges.append([a, b])
        max_node = max(max_node, a, b)

labels = {}
with open(LABEL_FILE) as f:
    for line in f:
        n, lab = map(int, line.split())
        labels[n] = lab
        max_node = max(max_node, n)

num_nodes = max_node + 1
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_index = to_undirected(edge_index)

y = torch.full((num_nodes,), -1, dtype=torch.long)
for n, lab in labels.items():
    y[n] = lab

# Features = degree + log-degree
deg = degree(edge_index[0], num_nodes=num_nodes).float()
x = torch.stack([deg, torch.log1p(deg)], dim=1)
x = (x - x.mean(0)) / (x.std(0) + 1e-9)

data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

# Splits
idxs = (y >= 0).nonzero(as_tuple=False).view(-1).tolist()
random.shuffle(idxs)
n = len(idxs)
train_idx, val_idx, test_idx = idxs[:int(0.6*n)], idxs[int(0.6*n):int(0.8*n)], idxs[int(0.8*n):]
train_mask = torch.zeros(num_nodes, dtype=torch.bool); train_mask[train_idx] = True
val_mask   = torch.zeros(num_nodes, dtype=torch.bool); val_mask[val_idx]   = True
test_mask  = torch.zeros(num_nodes, dtype=torch.bool); test_mask[test_idx] = True
data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

# Model
class MyGNN(MessagePassing):
    def __init__(self, in_ch, hidden_ch, out_ch, dropout=0.5):
        super().__init__(aggr="mean")
        self.lin1 = Linear(in_ch, hidden_ch)
        self.lin2 = Linear(hidden_ch, out_ch)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = F.relu(self.propagate(edge_index, x=x))
        x = self.dropout(x)
        return self.lin2(x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

# Training
model = MyGNN(data.num_node_features, 64, int(y.max().item())+1).to(DEVICE)
data = data.to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 201):
    model.train(); opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); opt.step()

    if epoch % 20 == 0:
        model.eval()
        pred = out.argmax(dim=1)
        train_acc = (pred[data.train_mask]==data.y[data.train_mask]).float().mean().item()
        val_acc   = (pred[data.val_mask]==data.y[data.val_mask]).float().mean().item()
        test_acc  = (pred[data.test_mask]==data.y[data.test_mask]).float().mean().item()
        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Train {train_acc:.3f} | Val {val_acc:.3f} | Test {test_acc:.3f}")
