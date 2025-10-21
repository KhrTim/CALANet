# SAGoG: Similarity-Aware Graph of Graphs Neural Networks

PyTorch implementation of SAGoG for Multivariate Time Series Classification.

## Overview

SAGoG is a neural network architecture that combines graph neural networks (GNNs) with a hierarchical "graph of graphs" approach for classifying multivariate time series. The model:

1. **Constructs graphs** from time series data using similarity measures (correlation, adaptive learning, or DTW)
2. **Encodes individual graphs** using GNN layers (GCN or GAT)
3. **Builds a meta-graph** where each node represents a time series window
4. **Classifies** based on the learned hierarchical graph representations

## Architecture Components

### 1. Graph Constructor
- **Correlation-based**: Uses Pearson correlation between variables
- **Adaptive**: Learns optimal graph structure via neural parameters
- **DTW-based**: Uses Dynamic Time Warping similarity

### 2. Graph-Level Encoder
- GCN (Graph Convolutional Networks) or GAT (Graph Attention Networks)
- Extracts features from individual time series graphs

### 3. Graph-of-Graphs Layer
- Constructs meta-graph where nodes are time series windows
- Applies GNN to learn relationships between windows

### 4. Models
- **SAGoG**: Full model with LSTM temporal encoding
- **SAGoGLite**: Lightweight version with Conv1D encoding

## Installation

```bash
pip install -r requirements.txt
```

### PyTorch Geometric Installation

If you encounter issues with `torch-geometric`, install it manually:

```bash
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

(Replace `cpu` with `cu118` for CUDA 11.8, `cu121` for CUDA 12.1, etc.)

## Quick Start

### Run Example

```bash
python example.py
```

This will run quick tests with synthetic data demonstrating:
- Basic SAGoG training
- Different graph construction methods
- SAGoGLite model
- Inference example

### Train on Synthetic Data

```bash
python train.py --dataset synthetic --num_samples 1000 --num_variables 10 --seq_len 100 --num_classes 3 --epochs 50
```

### Train with Different Configurations

```bash
# Use adaptive graph construction with GAT
python train.py --graph_construction adaptive --gnn_type gat --epochs 100

# Use correlation-based graphs
python train.py --graph_construction correlation --num_windows 7

# Use lightweight model
python train.py --model sagog_lite --hidden_dim 32 --batch_size 64
```

## Usage

### Basic Usage

```python
import torch
from sagog_model import SAGoG

# Create model
model = SAGoG(
    num_variables=10,      # Number of variables in time series
    seq_len=100,           # Length of time series
    num_classes=3,         # Number of classes
    hidden_dim=64,
    graph_hidden_dim=128,
    num_graph_layers=2,
    num_windows=5,
    graph_construction='adaptive',
    gnn_type='gcn'
)

# Input: [batch_size, num_variables, seq_len]
x = torch.randn(32, 10, 100)

# Forward pass
logits = model(x)  # Output: [32, 3]
predictions = torch.argmax(logits, dim=1)
```

### Training Loop

```python
from utils import train_epoch, evaluate, TimeSeriesDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Prepare data
train_dataset = TimeSeriesDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(100):
    train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch}: Accuracy = {train_metrics['accuracy']:.4f}")
```

## Command-Line Arguments

### Data Parameters
- `--dataset`: Dataset name (default: 'synthetic')
- `--num_samples`: Number of samples for synthetic data
- `--num_variables`: Number of variables
- `--seq_len`: Sequence length
- `--num_classes`: Number of classes

### Model Parameters
- `--model`: Model type ('sagog' or 'sagog_lite')
- `--hidden_dim`: Hidden dimension size
- `--graph_hidden_dim`: Graph encoder hidden dimension
- `--num_graph_layers`: Number of GNN layers
- `--num_windows`: Number of time series windows
- `--graph_construction`: Graph construction method ('correlation', 'adaptive', 'dtw')
- `--gnn_type`: GNN type ('gcn' or 'gat')

### Training Parameters
- `--batch_size`: Batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--weight_decay`: Weight decay for regularization
- `--patience`: Early stopping patience
- `--use_augmentation`: Enable data augmentation

## Model Variants

### SAGoG (Full Model)
- Uses bidirectional LSTM for temporal feature extraction
- More parameters but better performance
- Suitable for complex datasets

### SAGoGLite
- Uses Conv1D for temporal features
- Fewer parameters and faster training
- Suitable for resource-constrained settings

## Data Format

Input data should be in the format:
- **Shape**: `[num_samples, num_variables, seq_len]`
- **Type**: numpy array or torch tensor
- **Normalization**: Recommended to normalize each variable

Example:
```python
import numpy as np

# Generate sample data
num_samples = 1000
num_variables = 10
seq_len = 100

X = np.random.randn(num_samples, num_variables, seq_len)
y = np.random.randint(0, 3, size=num_samples)  # 3 classes
```

## Features

- ✅ Multiple graph construction methods
- ✅ Support for GCN and GAT
- ✅ Hierarchical graph-of-graphs architecture
- ✅ Data augmentation (jitter, scaling, time warping, window slicing)
- ✅ Early stopping and learning rate scheduling
- ✅ Model checkpointing
- ✅ Comprehensive evaluation metrics
- ✅ Synthetic data generation for testing

## Project Structure

```
SAGOG/
├── sagog_model.py      # Main model implementation
├── utils.py            # Utility functions (data loading, training, evaluation)
├── train.py            # Training script
├── example.py          # Usage examples
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Performance Tips

1. **Start with SAGoGLite** for initial experiments
2. **Adjust num_windows** based on your sequence length (typically 3-7)
3. **Use adaptive graph construction** for best results
4. **Enable data augmentation** for small datasets
5. **Monitor validation loss** and use early stopping
6. **Experiment with hidden dimensions** (32-128 typically works well)

## Citation

If you use this implementation, please cite the SAGoG paper:

```
SAGoG: Similarity-Aware Graph of Graphs Neural Networks for
Multivariate Time Series Classification
```

## License

This implementation is provided for research and educational purposes.