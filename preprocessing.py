import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load edges
edges = pd.read_csv('email-Eu-core.txt', sep=' ', header=None, names=['source', 'target'])

# Load labels
labels = pd.read_csv('email-Eu-core-department-labels.txt', sep=' ', header=None, names=['node', 'label'])

# Map node ids to consecutive integers
node_mapping = {node: i for i, node in enumerate(sorted(labels['node'].unique()))}

edges['source'] = edges['source'].map(node_mapping)
edges['target'] = edges['target'].map(node_mapping)
labels['node'] = labels['node'].map(node_mapping)

# using one-hot encoding of nodes
num_nodes = labels.shape[0]
x = torch.eye(num_nodes)  # Identity matrix as features

# using node degree as a feature
num_nodes = labels.shape[0]
degree = np.zeros(num_nodes)

for src, tgt in zip(edges['source'], edges['target']):
    degree[src] += 1
    degree[tgt] += 1

x = torch.tensor(degree, dtype=torch.float).unsqueeze(1)  # shape [num_nodes, 1]

# prepare edge index as PyTorch Geometric expects edges as a tensor of shape [2, num_edges]
edge_index = torch.tensor(edges[['source', 'target']].values.T, dtype=torch.long)

# prepare node labels
y = torch.tensor(labels.sort_values('node')['label'].values, dtype=torch.long)

# split data into train/validation/test
train_idx, test_idx = train_test_split(range(num_nodes), test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)  # 0.25*0.8=0.2

# create PyTorch Geometric Data Object
data = Data(x=x, edge_index=edge_index, y=y)

# optional masks
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.train_mask[train_idx] = True
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask[val_idx] = True
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask[test_idx] = True
