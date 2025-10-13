"""
Graph-Mamba inspired implementation for SNAP email-Eu-core
- Downloads SNAP email-Eu-core edge list
- Builds PyTorch Geometric Data
- Creates neighborhood tokenization + ordering
- Implements a lightweight Selective SSM-like encoder (Mamba-inspired)
- Trains a node classification model (department labels assumed available in a file)

Notes:
- This is an experimental, self-contained, PyG-based implementation inspired by Graph-Mamba.
- For the original paper and official code see:
  - https://arxiv.org/abs/2402.08678
  - https://github.com/bowang-lab/Graph-Mamba

Requirements:
pip install torch torch_geometric networkx scikit-learn requests tqdm

Run:
python graph_mamba_email_eu_core.py

"""
import os
import io
import math
import random
import requests
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_undirected
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------
# Utilities: download SNAP dataset
# -----------------------
SNAP_BASE = "https://snap.stanford.edu/data"
EU_CORE_URL = SNAP_BASE + "/email-Eu-core.txt.gz"
LABELS_URL = SNAP_BASE + "/email-Eu-core-department-labels.txt"  # hypothetical; SNAP provides labels in a separate file sometimes


def download_text_gz(url, local_path):
    if os.path.exists(local_path):
        return local_path
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_path


# -----------------------
# Build graph from edge list
# -----------------------

def build_graph_from_snap(local_gz_path):
    """Return NetworkX Graph and node list."""
    import gzip
    G = nx.DiGraph()
    with gzip.open(local_gz_path, 'rt') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'):
                continue
            u,v = line.split()
            u, v = int(u), int(v)
            G.add_edge(u, v)
    # relabel nodes to contiguous indices 0..n-1
    mapping = {n:i for i,n in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)
    return G


# -----------------------
# Feature engineering / preprocessing
# -----------------------

def compute_structural_features(G):
    # degree, clustering, pagerank
    n = G.number_of_nodes()
    deg = dict(G.degree())
    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())
    clustering = nx.clustering(G.to_undirected())
    try:
        pr = nx.pagerank(G)
    except Exception:
        pr = {i:0.0 for i in G.nodes()}

    X = torch.zeros((n, 5), dtype=torch.float32)
    for i in range(n):
        X[i,0] = deg.get(i, 0)
        X[i,1] = indeg.get(i, 0)
        X[i,2] = outdeg.get(i, 0)
        X[i,3] = clustering.get(i, 0)
        X[i,4] = pr.get(i, 0)
    # normalize
    sc = StandardScaler()
    X = torch.tensor(sc.fit_transform(X), dtype=torch.float32)
    return X


# -----------------------
# Neighborhood tokenization + ordering
# -----------------------

def tokenize_neighbors(G, max_tokens=32, ordering='degree'):
    """For each node produce a sequence of neighbor feature indices (including the center node).
    We return a dict: node -> list of neighbor node ids (length <= max_tokens).

    ordering: 'degree' (descending degree), 'random', 'pagerank'
    """
    pagerank = None
    if ordering == 'pagerank':
        try:
            pagerank = nx.pagerank(G)
        except Exception:
            pagerank = None

    neighbors_tokens = {}
    for v in G.nodes():
        nbrs = list(G.predecessors(v)) + list(G.successors(v))
        # unique and exclude self
        nbrs = [u for u in dict.fromkeys(nbrs) if u!=v]
        if ordering == 'degree':
            nbrs.sort(key=lambda u: G.degree(u), reverse=True)
        elif ordering == 'pagerank' and pagerank is not None:
            nbrs.sort(key=lambda u: pagerank.get(u,0), reverse=True)
        elif ordering == 'random':
            random.shuffle(nbrs)
        # include center node at position 0
        seq = [v] + nbrs[:max_tokens-1]
        neighbors_tokens[v] = seq
    return neighbors_tokens


# -----------------------
# Lightweight Selective SSM-like encoder (Mamba-inspired)
# We'll implement a simple bidirectional dilated depthwise conv stack with gating.
# This is NOT the official Mamba; it's an efficient approximation that captures long-range context.
# -----------------------


class SelectiveSSMEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel_size=3, layers=4, dilation_growth=2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        convs = []
        for i in range(layers):
            dilation = dilation_growth ** i
            padding = (kernel_size-1)*dilation//2
            convs.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation, groups=1))
            convs.append(nn.GELU())
            convs.append(nn.LayerNorm(hidden_dim))
        self.conv_net = nn.Sequential(*convs)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        h = self.input_proj(x)  # (B, L, H)
        # conv1d expects (B, C, L)
        h = h.permute(0,2,1)
        h = self.conv_net(h)
        h = h.permute(0,2,1)
        return self.out_proj(h)


# -----------------------
# Graph-Mamba inspired GNN: for each node, create neighborhood token sequence features -> SSM encoder -> readout
# -----------------------

class GraphMambaNodeClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, max_tokens=32):
        super().__init__()
        self.max_tokens = max_tokens
        self.local_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        # selective SSM-like encoder
        self.ssm = SelectiveSSMEncoder(hidden_dim, hidden_dim, layers=3)
        # readout
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, node_feat, neighbors_tokens):
        # node_feat: (N, F)
        device = node_feat.device
        N = node_feat.size(0)
        X = node_feat
        maxL = self.max_tokens
        H = self.local_encoder(X)  # (N, H)

        # Build batch sequences: for efficiency we process all nodes as a batch of sequences
        seq_feats = torch.zeros((N, maxL, H.size(1)), device=device)
        mask = torch.zeros((N, maxL), dtype=torch.bool, device=device)
        for v in range(N):
            seq = neighbors_tokens[v]
            L = len(seq)
            if L > maxL:
                seq = seq[:maxL]
                L = maxL
            # fetch features for nodes in seq
            seq_feats[v, :L] = H[torch.tensor(seq, device=device)]
            mask[v, :L] = 1
        # Apply SSM encoder
        out = self.ssm(seq_feats)  # (N, L, H)
        # Masked average pooling
        mask = mask.unsqueeze(-1)
        out = out * mask
        denom = mask.sum(1).clamp(min=1).float()
        pooled = out.sum(1) / denom  # (N, H)
        logits = self.classifier(pooled)
        return logits


# -----------------------
# Training & evaluation
# -----------------------


def train(model, optimizer, X, neighbors_tokens, y, train_idx):
    model.train()
    optimizer.zero_grad()
    logits = model(X, neighbors_tokens)
    loss = F.cross_entropy(logits[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, X, neighbors_tokens, y, idx):
    model.eval()
    with torch.no_grad():
        logits = model(X, neighbors_tokens)
        preds = logits.argmax(1)
        correct = (preds[idx] == y[idx]).sum().item()
        total = idx.size(0)
        acc = correct/total
    return acc


# -----------------------
# Main pipeline
# -----------------------

def main():
    # download data
    gz_path = 'email-Eu-core.txt.gz'
    if not os.path.exists(gz_path):
        print('Attempting to download email-Eu-core dataset...')
        download_text_gz(EU_CORE_URL, gz_path)

    G = build_graph_from_snap(gz_path)
    print('Nodes', G.number_of_nodes(), 'Edges', G.number_of_edges())

    # structural features
    X = compute_structural_features(G)

    # neighbors tokens
    max_tokens = 32
    neighbors_tokens = tokenize_neighbors(G, max_tokens=max_tokens, ordering='degree')

    # labels: SNAP provides department labels in a separate file on the page; here we try to fetch a labels file if available.
    # If not present, we create synthetic labels by clustering for demonstration.
    labels_path = 'email-Eu-core-department-labels.txt'
    y = None
    if os.path.exists(labels_path):
        # expected: node_id \t label per line
        lbl = []
        with open(labels_path,'r') as f:
            for line in f:
                a=line.strip().split()
                if not a: continue
                lbl.append(int(a[1]))
        y = torch.tensor(lbl, dtype=torch.long)
    else:
        print('Department labels file not found locally. Creating pseudo-labels with kmeans (demo only).')
        # create pseudo labels via spectral clustering on adjacency
        from sklearn.cluster import KMeans
        A = nx.to_scipy_sparse_matrix(G, format='csr')
        # compute simple spectral embedding (leading eigenvectors)
        try:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=16)
            emb = svd.fit_transform(A)
        except Exception:
            emb = X.numpy()
        k = 10
        kmeans = KMeans(n_clusters=k, random_state=42).fit(emb)
        y = torch.tensor(kmeans.labels_, dtype=torch.long)
        print('Using', k, 'pseudo-classes')

    N = X.size(0)
    num_classes = int(y.max().item()) + 1

    # create train/test split
    idx = list(range(N))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y.numpy())
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphMambaNodeClassifier(in_dim=X.size(1), hidden_dim=128, num_classes=num_classes, max_tokens=max_tokens).to(device)
    X = X.to(device)
    y = y.to(device)

    # prepare neighbors_tokens in CPU-friendly format (list per node)
    neighbors_tokens_list = {v: neighbors_tokens[v] for v in range(N)}

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_acc = 0.0
    for epoch in range(1, 201):
        loss = train(model, opt, X, neighbors_tokens_list, y, train_idx)
        if epoch % 10 == 0:
            acc = evaluate(model, X, neighbors_tokens_list, y, test_idx)
            if acc > best_acc:
                best_acc = acc
            print(f'Epoch {epoch:03d} Loss {loss:.4f} TestAcc {acc:.4f} Best {best_acc:.4f}')

    print('Training finished. Best test acc:', best_acc)


if __name__ == '__main__':
    main()
