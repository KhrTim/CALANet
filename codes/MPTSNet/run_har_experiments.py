"""
HAR Dataset Experiments using MPTSNet
Runs the same experiments as CALANet on HAR datasets
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
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Get the absolute paths before changing directory
current_dir = os.path.dirname(os.path.abspath(__file__))
codes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(codes_dir)

# Add MPTSNet path first so model can import its own utils
sys.path.insert(0, current_dir)
sys.path.insert(1, os.path.join(codes_dir, 'CALANet_local'))

# Import MPTSNet model and utilities BEFORE changing directory
from model.MPTSNet import Model
from utils import fft_main_periods_wo_duplicates

# Change to project root to access Data folder
os.chdir(project_root)

# Now import from CALANet utils (will be named calanet_utils to avoid conflicts)
import importlib.util
spec = importlib.util.spec_from_file_location("calanet_utils", os.path.join(codes_dir, 'CALANet_local/utils.py'))
calanet_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calanet_utils)
data_info = calanet_utils.data_info
Read_Data = calanet_utils.Read_Data
AvgrageMeter = calanet_utils.AvgrageMeter
accuracy = calanet_utils.accuracy

# Configuration
epoches = 25  # ULTRA-AGGRESSIVE: Reduced from 50 (KU-HAR still timed out at 60min with 50 epochs)
batch_size = 64  # Reduced to avoid memory issues
seed = 243
learning_rate = 0.001
weight_decay = 0.001
early_stopping_patience = 8  # Reduced from 10 for faster early stopping

# Dataset selection (uncomment the one you want to run)
#dataset = "UCI_HAR"
#dataset = "UniMiB-SHAR"
#dataset = "DSADS"
#dataset = "OPPORTUNITY"
#dataset = "KU-HAR"
#dataset = "PAMAP2"
dataset = "REALDISP"

input_nc, segment_size, class_num = data_info(dataset)

print(f"Running MPTSNet on {dataset}")
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

# Calculate adaptive embedding dimensions
num_channels = input_nc
seq_length = segment_size
embed_dim = max(min(num_channels * 4, 256), 64)
embed_dim_t = max(min(embed_dim * 4, 512), 256)

print(f"\nModel Configuration:")
print(f"  Adaptive embed_dim: {embed_dim}")
print(f"  Adaptive embed_dim_t: {embed_dim_t}")

# Detect periodic patterns using FFT on a sample of training data
print("\nDetecting periodic patterns using FFT...")
# Get a batch of training data for FFT analysis
sample_batch = next(iter(train_queue))
X_sample = sample_batch[0].float().to(device)

# Convert to format expected by FFT function (batch_size, seq_length, num_channels)
X_sample_fft = X_sample.permute(0, 2, 1).detach().cpu().numpy()

# Detect main periods
try:
    periods = fft_main_periods_wo_duplicates(X_sample_fft, 5, dataset)
    # Filter out periods that are too large or problematic
    periods = [int(p) for p in periods if int(p) > 1 and int(p) < seq_length]
    if not periods:
        # Use default periods if detection fails
        periods = [max(2, seq_length // 8), max(2, seq_length // 16), max(2, seq_length // 32)]
        periods = [p for p in periods if p > 1]
    periods = periods[:3]  # Use only top 3 periods to reduce computation
    print(f"Detected periods: {periods}")
except Exception as e:
    print(f"Period detection failed: {e}, using default periods")
    periods = [max(2, seq_length // 8), max(2, seq_length // 16), max(2, seq_length // 32)]
    periods = [p for p in periods if p > 1]

# Determine flag_DE_1 based on dataset
flag_DE_1 = False
if dataset in ['PAMAP2', 'OPPORTUNITY']:
    flag_DE_1 = True

# Create MPTSNet model
print("\nInitializing MPTSNet model...")
model = Model(
    periods=periods,
    flag=flag_DE_1,
    num_channels=num_channels,
    seq_length=seq_length,
    num_classes=class_num,
    embed_dim=embed_dim,
    embed_dim_t=embed_dim_t,
    num_heads=4,
    ff_dim=256,
    num_layers=1
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")

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
    weight_decay=weight_decay
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=0.0001
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

        # Handle NaN values
        x_train = torch.nan_to_num(x_train, nan=0.0)

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

            # Handle NaN values
            x = torch.nan_to_num(x, nan=0.0)

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
patience_counter = 0
lr_history = []

for epoch in range(epoches):
    # Training
    train_loss, train_acc = train(train_queue, model, criterion, optimizer)

    # Evaluation
    eval_loss, y_pred = infer(eval_queue, model, criterion)
    results = classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, zero_division=0)
    weighted_avg_f1 = results['weighted avg']['f1-score']

    # Update learning rate
    scheduler.step(eval_loss)

    # Track learning rate changes
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    if len(lr_history) > 1 and lr_history[-1] != lr_history[-2]:
        print(f"Epoch {epoch+1}: Learning rate changed to {current_lr}")

    if (epoch+1) % 50 == 0:
        print(f'Training... epoch {epoch+1}')

    if max_f1 < weighted_avg_f1:
        # Save best model
        os.makedirs('MPTSNet/save', exist_ok=True)
        torch.save(model.state_dict(), f'MPTSNet/save/{dataset}_mptsnet.pt')

        print(f"Epoch {epoch+1}, loss {eval_loss:.4e}, weighted f1 {weighted_avg_f1:.4f}, best_f1 {max_f1:.4f}")
        max_f1 = weighted_avg_f1
        best_epoch = epoch + 1
        patience_counter = 0

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
    else:
        patience_counter += 1

    # Early stopping check
    if patience_counter >= early_stopping_patience:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

print("\n" + "="*70)
print("Training Completed!")
print("="*70)
print(f"Best F1 Score: {max_f1:.4f} at epoch {best_epoch}")

# Load best model and evaluate
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(f'MPTSNet/save/{dataset}_mptsnet.pt'))
eval_loss, y_pred = infer(eval_queue, model, criterion)
y_pred_labels = np.argmax(y_pred, axis=1)

# Final metrics
results = classification_report(y_test_unary, y_pred_labels, digits=4, output_dict=True, zero_division=0)

print("\n" + "="*70)
print(f"MPTSNet Final Results on {dataset}")
print("="*70)
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1 (Weighted): {results['weighted avg']['f1-score']:.4f}")
print(f"Test F1 (Macro): {results['macro avg']['f1-score']:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test_unary, y_pred_labels, digits=4, zero_division=0))

# Save results to file
os.makedirs('MPTSNet/results', exist_ok=True)
with open(f'MPTSNet/results/{dataset}_mptsnet_results.txt', 'w') as f:
    f.write(f"MPTSNet Results on {dataset}\n")
    f.write("="*70 + "\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
    f.write(f"Test F1 (Weighted): {results['weighted avg']['f1-score']:.4f}\n")
    f.write(f"Test F1 (Macro): {results['macro avg']['f1-score']:.4f}\n")
    f.write("\nDetailed Classification Report:\n")
    f.write(classification_report(y_test_unary, y_pred_labels, digits=4, zero_division=0))

print(f"\nResults saved to MPTSNet/results/{dataset}_mptsnet_results.txt")
