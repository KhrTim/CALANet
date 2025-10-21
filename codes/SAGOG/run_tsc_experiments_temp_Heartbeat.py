"""
TSC Dataset Experiments using SAGoG
Runs experiments on Time Series Classification datasets
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from aeon.datasets import load_from_arff_file

# Get the absolute paths before changing directory
current_dir = os.path.dirname(os.path.abspath(__file__))
codes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(codes_dir)

# Add paths - CALANet utils first for priority
sys.path.insert(0, os.path.join(codes_dir, 'CALANet_local'))
sys.path.append(current_dir)

# Change to project root to access Data folder
os.chdir(project_root)

# Import from CALANet utils
from utils import AvgrageMeter, accuracy
from sagog_model import SAGoG

# Import EarlyStopping from SAGOG's utils
import importlib.util
spec = importlib.util.spec_from_file_location("sagog_utils", os.path.join(current_dir, "utils.py"))
sagog_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sagog_utils)
EarlyStopping = sagog_utils.EarlyStopping

# Configuration
epoches = 200
batch_size = 16  # Reduced from 64 to avoid OOM
seed = 243
learning_rate = 5e-4
weight_decay = 5e-4
early_stopping_patience = 30

# Dataset selection (uncomment the one you want to run)
#dataset = "AtrialFibrillation"
#dataset = "MotorImagery"
dataset = "Heartbeat"
#dataset = "PhonemeSpectra"
#dataset = "LSST"
#dataset = "PEMS-SF"

# TSC dataset information
TSC_INFO = {
    "AtrialFibrillation": {"channels": 2, "length": 640, "classes": 3},
    "MotorImagery": {"channels": 64, "length": 3000, "classes": 2},
    "Heartbeat": {"channels": 61, "length": 405, "classes": 2},
    "PhonemeSpectra": {"channels": 11, "length": 217, "classes": 39},
    "LSST": {"channels": 6, "length": 36, "classes": 14},
    "PEMS-SF": {"channels": 963, "length": 144, "classes": 7}
}

input_nc = TSC_INFO[dataset]["channels"]
segment_size = TSC_INFO[dataset]["length"]
class_num = TSC_INFO[dataset]["classes"]

print(f"Running SAGoG on {dataset}")
print(f"Input channels: {input_nc}, Segment size: {segment_size}, Classes: {class_num}")

# Set random seeds
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    device = 'cuda'
else:
    device = 'cpu'
    print('No GPU available, using CPU')

# Load TSC data
DATA_PATH = os.path.join('Data', 'TSC', dataset)
train_X, train_Y = load_from_arff_file(os.path.join(DATA_PATH, dataset + "_TRAIN.arff"))
_, train_Y = np.unique(train_Y, return_inverse=True)

test_X, test_Y = load_from_arff_file(os.path.join(DATA_PATH, dataset + "_TEST.arff"))
_, test_Y = np.unique(test_Y, return_inverse=True)

print(f"Train shape: {train_X.shape}, Test shape: {test_X.shape}")

# Create datasets
train_data = TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_Y))
test_data = TensorDataset(torch.FloatTensor(test_X), torch.LongTensor(test_Y))

# Create data loaders
train_queue = DataLoader(
    train_data, batch_size=batch_size, shuffle=True,
    pin_memory=True, num_workers=0)
eval_queue = DataLoader(
    test_data, batch_size=batch_size, shuffle=False,
    pin_memory=True, num_workers=0)

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# Create SAGoG model
print("\nInitializing SAGoG model...")

# Adjust number of windows based on sequence length and channel count
# Reduce windows for high-channel datasets to save memory
if input_nc >= 50:
    # For high channel count (like Heartbeat with 61 channels), use fewer windows
    num_windows = 3
elif segment_size >= 1000:
    num_windows = 10
elif segment_size >= 500:
    num_windows = 8
elif segment_size >= 200:
    num_windows = 5
else:
    num_windows = 3

# Adjust hidden dimensions based on number of channels
# Reduce dimensions to avoid OOM
if input_nc >= 500:
    hidden_dim = 16
    graph_hidden_dim = 32
    num_graph_layers = 1
elif input_nc >= 50:  # Heartbeat has 61 channels
    hidden_dim = 32
    graph_hidden_dim = 64
    num_graph_layers = 1
else:
    hidden_dim = 64
    graph_hidden_dim = 128
    num_graph_layers = 2

model = SAGoG(
    num_variables=input_nc,
    seq_len=segment_size,
    num_classes=class_num,
    hidden_dim=hidden_dim,
    graph_hidden_dim=graph_hidden_dim,
    num_graph_layers=num_graph_layers,
    num_windows=num_windows,
    graph_construction='adaptive',
    gnn_type='gcn'
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Number of windows: {num_windows}")
print(f"Hidden dim: {hidden_dim}, Graph hidden dim: {graph_hidden_dim}")

# Weight initialization
def weight_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.01)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(weight_init)

# Create optimizer and loss
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    eps=1e-08
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# Early stopping
early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')

# Training function
def train(train_queue, model, criterion, optimizer):
    cl_loss = AvgrageMeter()
    cl_acc = AvgrageMeter()
    model.train()

    for step, (x_train, y_train) in enumerate(train_queue):
        n = x_train.size(0)
        x_train = x_train.to(device).float()
        y_train = y_train.to(device).long()

        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)

        loss.backward()
        optimizer.step()

        prec1 = accuracy(logits.cpu().detach(), y_train.cpu())
        cl_loss.update(loss.data.item(), n)
        cl_acc.update(prec1, n)

    return cl_loss.avg, cl_acc.avg

# Inference function
def infer(eval_queue, model, criterion):
    cl_loss = AvgrageMeter()
    model.eval()

    preds = []
    with torch.no_grad():
        for step, (x, y) in enumerate(eval_queue):
            x = x.to(device).float()
            y = y.to(device).long()

            logits = model(x)
            loss = criterion(logits, y)
            preds.extend(logits.cpu().numpy())

            n = x.size(0)
            cl_loss.update(loss.data.item(), n)

    return cl_loss.avg, np.asarray(preds)

# Training loop
print("\nStarting training...")
print("="*70)

max_acc = 0
best_acc = 0
best_epoch = 0

for epoch in range(epoches):
    # Training
    train_loss, train_acc = train(train_queue, model, criterion, optimizer)

    # Evaluation
    eval_loss, y_pred = infer(eval_queue, model, criterion)
    y_pred_labels = np.argmax(y_pred, axis=1)
    test_acc = accuracy_score(test_Y, y_pred_labels)

    # Update learning rate
    scheduler.step(eval_loss)

    if (epoch+1) % 50 == 0:
        print(f'Training... epoch {epoch+1}')

    if max_acc < test_acc:
        # Save best model
        os.makedirs('codes/SAGOG/save', exist_ok=True)
        torch.save(model.state_dict(), f'codes/SAGOG/save/{dataset}_sagog.pt')

        print(f"Epoch {epoch+1}, loss {eval_loss:.4e}, accuracy {test_acc:.4f}, best_acc {max_acc:.4f}")
        max_acc = test_acc
        best_epoch = epoch + 1

        print(classification_report(test_Y, y_pred_labels, digits=4, zero_division=0))

    # Early stopping check
    if early_stopping(test_acc):
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

print("\n" + "="*70)
print("Training Completed!")
print("="*70)
print(f"Best Accuracy: {max_acc:.4f} at epoch {best_epoch}")

# Load best model and evaluate
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(f'codes/SAGOG/save/{dataset}_sagog.pt'))
eval_loss, y_pred = infer(eval_queue, model, criterion)
y_pred_labels = np.argmax(y_pred, axis=1)

# Final metrics
results = classification_report(test_Y, y_pred_labels, digits=4, output_dict=True, zero_division=0)

print("\n" + "="*70)
print(f"SAGoG Final Results on {dataset}")
print("="*70)
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1 (Weighted): {results['weighted avg']['f1-score']:.4f}")
print(f"Test F1 (Macro): {results['macro avg']['f1-score']:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(test_Y, y_pred_labels, digits=4, zero_division=0))

# Save results to file
os.makedirs('codes/SAGOG/results', exist_ok=True)
with open(f'codes/SAGOG/results/{dataset}_sagog_results.txt', 'w') as f:
    f.write(f"SAGoG Results on {dataset}\n")
    f.write("="*70 + "\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
    f.write(f"Test F1 (Weighted): {results['weighted avg']['f1-score']:.4f}\n")
    f.write(f"Test F1 (Macro): {results['macro avg']['f1-score']:.4f}\n")
    f.write("\nDetailed Classification Report:\n")
    f.write(classification_report(test_Y, y_pred_labels, digits=4, zero_division=0))

print(f"\nResults saved to codes/SAGOG/results/{dataset}_sagog_results.txt")
