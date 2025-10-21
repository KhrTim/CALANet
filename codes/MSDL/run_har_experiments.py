"""
HAR Dataset Experiments using MSDL
Runs the same experiments as CALANet on HAR datasets
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Get the absolute paths before changing directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add paths
sys.path.append(os.path.join(parent_dir, 'codes/CALANet_local'))
sys.path.append(current_dir)

# Change to parent directory to access Data folder
os.chdir(parent_dir)

from utils import data_info, Read_Data, AvgrageMeter, accuracy
from msdl import MSDL
from train import MSDLTrainer

# Configuration
epoches = 500
batch_size = 128
seed = 243
learning_rate = 5e-4
weight_decay = 5e-4

# Dataset selection (uncomment the one you want to run)
#dataset = "UCI_HAR"  # Default dataset
#dataset = "UniMiB-SHAR"
#dataset = "DSADS"
#dataset = "OPPORTUNITY"
#dataset = "KU-HAR"
#dataset = "PAMAP2"
dataset = "REALDISP"

input_nc, segment_size, class_num = data_info(dataset)

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

# Load data
train_data, eval_data, y_test_unary = Read_Data(dataset, input_nc)

# Create data loaders
train_queue = DataLoader(
    train_data, batch_size=batch_size, shuffle=True,
    pin_memory=True, num_workers=0)
eval_queue = DataLoader(
    eval_data, batch_size=batch_size, shuffle=False,
    pin_memory=True, num_workers=0)

print(f"Train samples: {len(train_data)}, Test samples: {len(eval_data)}")

# Create MSDL model
print("\nInitializing MSDL model...")
model = MSDL(
    input_channels=input_nc,
    num_classes=class_num,
    multiscale_channels=64,
    kernel_sizes=[3, 5, 7, 9],  # Multiple scales
    lstm_hidden=128,
    lstm_layers=2,
    dropout=0.5
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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

max_f1 = 0
weighted_avg_f1 = 0
best_epoch = 0

for epoch in range(epoches):
    # Training
    train_loss, train_acc = train(train_queue, model, criterion, optimizer)

    # Evaluation
    eval_loss, y_pred = infer(eval_queue, model, criterion)
    results = classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, zero_division=0)
    weighted_avg_f1 = results['weighted avg']['f1-score']

    if (epoch+1) % 50 == 0:
        print(f'Training... epoch {epoch+1}')

    if max_f1 < weighted_avg_f1:
        # Save best model
        os.makedirs('MSDL/save', exist_ok=True)
        torch.save(model.state_dict(), f'MSDL/save/{dataset}_msdl.pt')

        print(f"Epoch {epoch+1}, loss {eval_loss:.4e}, weighted f1 {weighted_avg_f1:.4f}, best_f1 {max_f1:.4f}")
        max_f1 = weighted_avg_f1
        best_epoch = epoch + 1

        print(classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, zero_division=0))

        # Dataset-specific metrics (matching CALANet output)
        if dataset == 'UniMiB-SHAR':
            adl_f1 = classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=list(range(9)), zero_division=0)['weighted avg']['f1-score']
            falls_f1 = classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=list(range(9,17)), zero_division=0)['weighted avg']['f1-score']
            print(f'ADL: {adl_f1:.4f}')
            print(f'Falls: {falls_f1:.4f}')
        elif dataset == 'PAMAP2':
            adl_f1 = classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=[0,1,2,3,4,5,6,10,11,12,13,17], zero_division=0)['weighted avg']['f1-score']
            complex_f1 = classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=[7,8,9,14,15,16], zero_division=0)['weighted avg']['f1-score']
            print(f'ADL: {adl_f1:.4f}')
            print(f'Complex: {complex_f1:.4f}')

print("\n" + "="*70)
print("Training Completed!")
print("="*70)
print(f"Best F1 Score: {max_f1:.4f} at epoch {best_epoch}")

# Load best model and evaluate
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(f'MSDL/save/{dataset}_msdl.pt'))
eval_loss, y_pred = infer(eval_queue, model, criterion)
y_pred_labels = np.argmax(y_pred, axis=1)

# Final metrics
results = classification_report(y_test_unary, y_pred_labels, digits=4, output_dict=True, zero_division=0)

print("\n" + "="*70)
print(f"MSDL Final Results on {dataset}")
print("="*70)
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1 (Weighted): {results['weighted avg']['f1-score']:.4f}")
print(f"Test F1 (Macro): {results['macro avg']['f1-score']:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test_unary, y_pred_labels, digits=4, zero_division=0))

# Save results to file
os.makedirs('MSDL/results', exist_ok=True)
with open(f'MSDL/results/{dataset}_msdl_results.txt', 'w') as f:
    f.write(f"MSDL Results on {dataset}\n")
    f.write("="*70 + "\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
    f.write(f"Test F1 (Weighted): {results['weighted avg']['f1-score']:.4f}\n")
    f.write(f"Test F1 (Macro): {results['macro avg']['f1-score']:.4f}\n")
    f.write("\nDetailed Classification Report:\n")
    f.write(classification_report(y_test_unary, y_pred_labels, digits=4, zero_division=0))

print(f"\nResults saved to MSDL/results/{dataset}_msdl_results.txt")
