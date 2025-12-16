"""
TSC Dataset Experiments using MSDL
Runs experiments on Time Series Classification datasets
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
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

# Remove script directory from sys.path if it was auto-added by Python
if current_dir in sys.path:
    sys.path.remove(current_dir)

# Add paths - CALANet utils first for priority
sys.path.insert(0, os.path.join(codes_dir, 'CALANet_local'))
sys.path.append(current_dir)

# Change to project root to access Data folder
os.chdir(project_root)

from utils import AvgrageMeter, accuracy
from msdl import MSDL

# Import shared metrics collector
import importlib.util
codes_dir_for_metrics = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("shared_metrics",
                                              os.path.join(codes_dir_for_metrics, 'shared_metrics.py'))
shared_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_metrics)
MetricsCollector = shared_metrics.MetricsCollector


# Configuration
epoches = 100  # Reduced from 200 to prevent timeouts
batch_size = 64
seed = 243
learning_rate = 5e-4
weight_decay = 5e-4

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

print(f"Running MSDL on {dataset}")
print(f"Input channels: {input_nc}, Segment size: {segment_size}, Classes: {class_num}")

# Set random seeds
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
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

# Create MSDL model
print("\nInitializing MSDL model...")

# Adjust multiscale_channels based on number of input channels
if input_nc >= 500:
    multiscale_channels = 32
    lstm_hidden = 64
elif input_nc >= 100:
    multiscale_channels = 48
    lstm_hidden = 96
else:
    multiscale_channels = 64
    lstm_hidden = 128

model = MSDL(
    input_channels=input_nc,
    num_classes=class_num,
    multiscale_channels=multiscale_channels,
    kernel_sizes=[3, 5, 7, 9],  # Multiple scales
    lstm_hidden=lstm_hidden,
    lstm_layers=2,
    dropout=0.5
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Multiscale channels: {multiscale_channels}, LSTM hidden: {lstm_hidden}")

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

# Initialize metrics collector
metrics_collector = MetricsCollector(
    model_name='MSDL',
    dataset=dataset,
    task_type='TSC',
    save_dir='results'
)

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

# Track training time
with metrics_collector.track_training():
    for epoch in range(epoches):
        # Training
        train_loss, train_acc = train(train_queue, model, criterion, optimizer)

        # Evaluation
        eval_loss, y_pred = infer(eval_queue, model, criterion)
        y_pred_labels = np.argmax(y_pred, axis=1)
        test_acc = accuracy_score(test_Y, y_pred_labels)

        if (epoch+1) % 50 == 0:
            print(f'Training... epoch {epoch+1}')

        if max_acc < test_acc:
            # Save best model
            os.makedirs('codes/MSDL/save', exist_ok=True)
            torch.save(model.state_dict(), f'codes/MSDL/save/{dataset}_msdl.pt')

            print(f"Epoch {epoch+1}, loss {eval_loss:.4e}, accuracy {test_acc:.4f}, best_acc {max_acc:.4f}")
            max_acc = test_acc
            best_epoch = epoch + 1

            print(classification_report(test_Y, y_pred_labels, digits=4, zero_division=0))

print("\n" + "="*70)
print("Training Completed!")
print("="*70)
print(f"Best Accuracy: {max_acc:.4f} at epoch {best_epoch}")

# Load best model and evaluate
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(f'codes/MSDL/save/{dataset}_msdl.pt'))
eval_loss, y_pred = infer(eval_queue, model, criterion)
y_pred_labels = np.argmax(y_pred, axis=1)

# Final metrics
results = classification_report(test_Y, y_pred_labels, digits=4, output_dict=True, zero_division=0)

print("\n" + "="*70)
print(f"MSDL Final Results on {dataset}")
print("="*70)
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1 (Weighted): {results['weighted avg']['f1-score']:.4f}")
print(f"Test F1 (Macro): {results['macro avg']['f1-score']:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(test_Y, y_pred_labels, digits=4, zero_division=0))

# Save results to file
os.makedirs('codes/MSDL/results', exist_ok=True)
with open(f'codes/MSDL/results/{dataset}_msdl_results.txt', 'w') as f:
    f.write(f"MSDL Results on {dataset}\n")
    f.write("="*70 + "\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
    f.write(f"Test F1 (Weighted): {results['weighted avg']['f1-score']:.4f}\n")
    f.write(f"Test F1 (Macro): {results['macro avg']['f1-score']:.4f}\n")
    f.write("\nDetailed Classification Report:\n")
    f.write(classification_report(test_Y, y_pred_labels, digits=4, zero_division=0))

print(f"\nResults saved to codes/MSDL/results/{dataset}_msdl_results.txt")


# ============================================================================
# COMPREHENSIVE METRICS COLLECTION
# ============================================================================
print("\n" + "="*70)
print("COLLECTING COMPREHENSIVE METRICS")
print("="*70)

# Track inference time
with metrics_collector.track_inference():
    # Re-run inference for timing
    if 'eval_queue' in locals():
        eval_loss, y_pred = infer(eval_queue, model, criterion)
    elif 'test_queue' in locals():
        eval_loss, y_pred = infer(test_queue, model, criterion)
    else:
        y_pred = model(X_test_torch if 'X_test_torch' in locals() else torch.FloatTensor(X_test).to(device))

# Compute throughput
test_samples = len(y_test_unary) if 'y_test_unary' in locals() else (len(test_Y) if 'test_Y' in locals() else (len(y_test) if 'y_test' in locals() else len(eval_data)))
metrics_collector.compute_throughput(test_samples, phase='inference')

# Compute classification metrics
if hasattr(y_pred, 'cpu'):
    y_pred_np = y_pred.cpu().numpy() if hasattr(y_pred, 'cpu') else y_pred
else:
    y_pred_np = y_pred

y_pred_labels = np.argmax(y_pred_np, axis=1) if len(y_pred_np.shape) > 1 else y_pred_np

y_true_labels = y_test_unary if 'y_test_unary' in locals() else (test_Y if 'test_Y' in locals() else y_test)
metrics_collector.compute_classification_metrics(y_true_labels, y_pred_labels)

# Compute model complexity
input_shape = (1, input_nc, segment_size)
if input_shape is not None:
    try:
        metrics_collector.compute_model_complexity(model, input_shape, device=device if 'device' in locals() else 'cuda')
    except Exception as e:
        print(f"Could not compute model complexity: {e}")

# Save comprehensive metrics
metrics_collector.save_metrics()
metrics_collector.print_summary()

