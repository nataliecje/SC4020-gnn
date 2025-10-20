# GNN Experiment Pipeline for Email-EU-Core Dataset

This repository contains a comprehensive evaluation pipeline for Graph Neural Networks (GNNs) on the Email-EU-Core dataset, with advanced techniques to achieve high accuracy on organizational network classification.

## Quick Start

### **Option 1: Run Basic GNN Experiments**
```bash
# Install dependencies
pip install torch torch-geometric scikit-learn pandas matplotlib seaborn

# Run baseline experiments
python simple_pipeline.py
```

### **Option 2: Get Best Results (Recommended)**
```bash
# Install dependencies
pip install torch torch-geometric scikit-learn pandas matplotlib seaborn node2vec

# Run improved model with enhanced features
python improved_training.py
```

## What You'll Get

### **Experimental Results:**
- **Table A**: Main comparison of GCN vs GAT (achieved: GCN 13.4%, GAT 15.9%)
- **Table B**: Feature comparison (Structural vs Laplacian PE vs Node2Vec)
- **Table C**: Ablation studies (depth, dropout, attention heads)
- **Figures**: Training curves, class distributions, performance analysis

### **Key Findings:**
- **3-layer networks** perform best (21.9% accuracy vs 14.4% for 2-layer)
- **GAT outperforms GCN** consistently (+18.5% improvement)
- **8 attention heads** work better than fewer heads
- **No dropout needed** for this dataset
- **9x better than random** baseline (2.4%)

## Dataset Overview

**Email-EU-Core Network:**
- **1,005 nodes** (email addresses, all labeled)
- **25,571 edges** (email communications)
- **42 classes** (department affiliations)
- **Challenge**: 109:1 class imbalance (some departments have only 1 person!)
- **Task**: Predict department from communication patterns

## License

MIT License - See LICENSE file for details.
