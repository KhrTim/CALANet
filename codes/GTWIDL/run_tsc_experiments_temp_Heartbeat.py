"""
TSC Dataset Experiments using GTWIDL
Runs experiments on Time Series Classification datasets
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import sys
from sklearn.metrics import classification_report, accuracy_score, f1_score
from aeon.datasets import load_from_arff_file

# Get the absolute paths before changing directory
current_dir = os.path.dirname(os.path.abspath(__file__))
codes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(codes_dir)

# Add paths
sys.path.insert(0, os.path.join(codes_dir, 'CALANet_local'))
sys.path.append(current_dir)

# Change to project root to access Data folder
os.chdir(project_root)

from gtwidl import GTWIDL
from classification import GTWIDLClassifier

# Configuration
seed = 243
n_atoms = 20
atom_length_ratio = 1.0  # Use full length for TSC datasets

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

# Load TSC data
DATA_PATH = os.path.join('Data', 'TSC', dataset)
train_X, train_Y = load_from_arff_file(os.path.join(DATA_PATH, dataset + "_TRAIN.arff"))
_, train_Y = np.unique(train_Y, return_inverse=True)

test_X, test_Y = load_from_arff_file(os.path.join(DATA_PATH, dataset + "_TEST.arff"))
_, test_Y = np.unique(test_Y, return_inverse=True)

print(f"Train shape: {train_X.shape}, Test shape: {test_X.shape}")

# GTWIDL expects input shape: (N, L, C) where L is time and C is channels
# Need to transpose from (N, C, L) to (N, L, C)
X_train = np.transpose(train_X, (0, 2, 1))  # (N, L, C)
X_test = np.transpose(test_X, (0, 2, 1))
y_train = train_Y
y_test = test_Y

print(f"After transposing - Train: {X_train.shape}, Test: {X_test.shape}")

# atom_length should equal series_length for proper warping
atom_length = segment_size
print(f"Atom length: {atom_length}")

# Convert to torch tensors
X_train_torch = torch.FloatTensor(X_train).to(device)
X_test_torch = torch.FloatTensor(X_test).to(device)

# Create GTWIDL model
print("\nTraining GTWIDL Dictionary...")
gtwidl_model = GTWIDL(
    n_atoms=n_atoms,
    atom_length=atom_length,
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
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')
test_acc = accuracy_score(y_test, y_pred)

# Print results
print("\n" + "="*70)
print(f"GTWIDL Results on {dataset}")
print("="*70)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 (Weighted): {f1_weighted:.4f}")
print(f"Test F1 (Macro): {f1_macro:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Save results to file
os.makedirs('codes/GTWIDL/results', exist_ok=True)
with open(f'codes/GTWIDL/results/{dataset}_gtwidl_results.txt', 'w') as f:
    f.write(f"GTWIDL Results on {dataset}\n")
    f.write("="*70 + "\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test F1 (Weighted): {f1_weighted:.4f}\n")
    f.write(f"Test F1 (Macro): {f1_macro:.4f}\n")
    f.write("\nDetailed Classification Report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))

print(f"\nResults saved to codes/GTWIDL/results/{dataset}_gtwidl_results.txt")
