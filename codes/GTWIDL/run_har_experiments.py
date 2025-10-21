"""
HAR Dataset Experiments using GTWIDL
Runs the same experiments as CALANet on HAR datasets
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import sys
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Get the absolute paths before changing directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add paths - CALANet utils first for priority
sys.path.insert(0, os.path.join(parent_dir, 'codes/CALANet_local'))
sys.path.append(current_dir)

# Change to parent directory to access Data folder
os.chdir(parent_dir)

from utils import data_info, Read_Data
from gtwidl import GTWIDL
from classification import GTWIDLClassifier

# Configuration
epoches = 100
batch_size = 128
seed = 243
n_atoms = 20
atom_length_ratio = 0.5  # atom length as ratio of segment size

# Dataset selection (uncomment the one you want to run)
#dataset = "UCI_HAR"
#dataset = "UniMiB-SHAR"
dataset = "DSADS"
#dataset = "OPPORTUNITY"
#dataset = "KU-HAR"
#dataset = "PAMAP2"
#dataset = "REALDISP"

input_nc, segment_size, class_num = data_info(dataset)

print(f"Running GTWIDL on {dataset}")
print(f"Input channels: {input_nc}, Segment size: {segment_size}, Classes: {class_num}")
print(f"Number of atoms: {n_atoms}")

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

# Extract numpy arrays from datasets
X_train = train_data.x.numpy()  # Shape: (N, C, L)
y_train = train_data.y.numpy()
X_test = eval_data.x.numpy()
y_test = eval_data.y.numpy()

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# GTWIDL expects input shape: (N, L, C) where L is time and C is channels
# Need to transpose from (N, C, L) to (N, L, C)
X_train = np.transpose(X_train, (0, 2, 1))  # (N, L, C)
X_test = np.transpose(X_test, (0, 2, 1))

print(f"After transposing - Train: {X_train.shape}, Test: {X_test.shape}")

# atom_length should equal series_length for proper warping
# The warping will handle time alignment, so atoms should be same length as data
atom_length = segment_size
print(f"Atom length: {atom_length}")

# Convert to torch tensors
X_train_torch = torch.FloatTensor(X_train).to(device)
X_test_torch = torch.FloatTensor(X_test).to(device)

# Create GTWIDL model
print("\nTraining GTWIDL Dictionary...")
gtwidl_model = GTWIDL(
    n_atoms=n_atoms,
    atom_length=atom_length,  # Use adjusted atom_length
    n_basis=5,
    basis_type='polynomial',
    lambda_sparse=0.1,
    max_iter=50,
    device=device,
    verbose=True
)

# Train dictionary (this might take a while for large datasets)
# For very large datasets, consider using a subset for dictionary learning
max_samples_for_dict = 5000
if len(X_train_torch) > max_samples_for_dict:
    print(f"\nUsing {max_samples_for_dict} samples for dictionary learning (full dataset is {len(X_train_torch)})")
    indices = np.random.choice(len(X_train_torch), max_samples_for_dict, replace=False)
    X_train_dict = X_train_torch[indices]
else:
    X_train_dict = X_train_torch

dictionary, alphas_train, betas_train = gtwidl_model.fit(X_train_dict)

print("\nDictionary learning completed!")

# Create classifier
print("\nTraining SVM Classifier on GTWIDL features...")
classifier = GTWIDLClassifier(
    gtwidl_model=gtwidl_model,
    classifier_type='svm',
    classifier_params={'random_state': seed}
)

# Train classifier on full training set
classifier.fit(X_train_torch, y_train)

# Evaluate
print("\nEvaluating on test set...")
test_metrics = classifier.evaluate(X_test_torch, y_test, verbose=False)

# Get predictions for detailed metrics
y_pred = classifier.predict(X_test_torch)

# Compute additional metrics
from sklearn.metrics import f1_score
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

# Print results
print("\n" + "="*70)
print(f"GTWIDL Results on {dataset}")
print("="*70)
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test F1 (Weighted): {f1_weighted:.4f}")
print(f"Test F1 (Macro): {f1_macro:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Dataset-specific metrics (matching CALANet output)
if dataset == 'UniMiB-SHAR':
    adl_f1 = classification_report(y_test, y_pred, digits=4, output_dict=True, labels=list(range(9)))['weighted avg']['f1-score']
    falls_f1 = classification_report(y_test, y_pred, digits=4, output_dict=True, labels=list(range(9,17)))['weighted avg']['f1-score']
    print(f'ADL: {adl_f1:.4f}')
    print(f'Falls: {falls_f1:.4f}')
elif dataset == 'PAMAP2':
    adl_f1 = classification_report(y_test, y_pred, digits=4, output_dict=True, labels=[0,1,2,3,4,5,6,10,11,12,13,17])['weighted avg']['f1-score']
    complex_f1 = classification_report(y_test, y_pred, digits=4, output_dict=True, labels=[7,8,9,14,15,16])['weighted avg']['f1-score']
    print(f'ADL: {adl_f1:.4f}')
    print(f'Complex: {complex_f1:.4f}')

# Save results to file
os.makedirs('GTWIDL/results', exist_ok=True)
with open(f'GTWIDL/results/{dataset}_gtwidl_results.txt', 'w') as f:
    f.write(f"GTWIDL Results on {dataset}\n")
    f.write("="*70 + "\n")
    f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
    f.write(f"Test F1 (Weighted): {f1_weighted:.4f}\n")
    f.write(f"Test F1 (Macro): {f1_macro:.4f}\n")
    f.write("\nDetailed Classification Report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))

print(f"\nResults saved to GTWIDL/results/{dataset}_gtwidl_results.txt")
