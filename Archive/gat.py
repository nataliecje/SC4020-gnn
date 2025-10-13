import os, random, math
import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
from torch_geometric.nn import GATConv

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EDGE_FILE = "email-Eu-core.txt"
LABEL_FILE = "email-Eu-core-department-labels.txt"

# Load data
edges, labels, max_node = [], {}, -1
with open(EDGE_FILE) as f:
    for line in f:
        a, b = map(int, line.split()); edges.append([a, b]); max_node = max(max_node, a, b)
with open(LABEL_FILE) as f:
    for line in f:
        n, lab = map(int, line.split()); labels[n] = lab; max_node = max(max_node, n)
num_nodes = max_node+1
edge_index = torch.tensor(edges).t().contiguous()
edge_index = to_undirected(edge_index)
y = torch.full((num_nodes,), -1, dtype=torch.long)
for n, lab in labels.items(): y[n] = lab

deg = degree(edge_index[0], num_nodes=num_nodes).float()
x = torch.stack([deg, torch.log1p(deg)], dim=1)
x = (x - x.mean(0)) / (x.std(0) + 1e-9)
data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

# Splits
idxs = (y>=0).nonzero(as_tuple=False).view(-1).tolist()
random.shuffle(idxs); n = len(idxs)
train_idx, val_idx, test_idx = idxs[:int(.6*n)], idxs[int(.6*n):int(.8*n)], idxs[int(.8*n):]
train_mask = torch.zeros(num_nodes, dtype=torch.bool); train_mask[train_idx]=True
val_mask = torch.zeros(num_nodes, dtype=torch.bool); val_mask[val_idx]=True
test_mask = torch.zeros(num_nodes, dtype=torch.bool); test_mask[test_idx]=True
data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

# Model
class GAT(torch.nn.Module):
    def __init__(self, in_ch, hidden, out_ch, heads=4, dropout=0.5):
        super().__init__()
        self.gat1 = GATConv(in_ch, hidden, heads=heads)
        self.gat2 = GATConv(hidden*heads, out_ch, heads=1, concat=False)
        self.dropout = Dropout(dropout)
    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        return self.gat2(x, edge_index)

model = GAT(data.num_node_features, 32, int(y.max())+1).to(DEVICE)
data = data.to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

for epoch in range(1, 201):
    model.train(); opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); opt.step()
    if epoch % 20 == 0:
        model.eval(); pred = out.argmax(dim=1)
        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
              f"Train {(pred[data.train_mask]==data.y[data.train_mask]).float().mean():.3f} | "
              f"Val {(pred[data.val_mask]==data.y[data.val_mask]).float().mean():.3f} | "
              f"Test {(pred[data.test_mask]==data.y[data.test_mask]).float().mean():.3f}")
