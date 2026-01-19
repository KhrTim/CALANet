#!/usr/bin/env python3
"""
Evaluate original checkpoints from RTHAR.zip to see if they produce paper results
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score

# Add codes directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'codes'))

from paper_results import PAPER_HAR_F1

# Original checkpoint path - try both locations
ORIGINAL_CHECKPOINT_DIR = "/tmp/original_checkpoints/codes/CALANet_local/save/with_gts"
CURRENT_CHECKPOINT_DIR = "HT-AggNet_v2/save/with_gts"

# HAR datasets to test
HAR_DATASETS = ["UCI_HAR", "DSADS", "OPPORTUNITY", "KU-HAR", "PAMAP2", "REALDISP"]

def load_har_data(dataset):
    """Load HAR dataset"""
    # Data is in Data/preprocessed/{dataset}/ relative to project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, 'Data', 'preprocessed', dataset)

    x_train = np.load(os.path.join(data_path, 'train_x.npy'))
    y_train = np.load(os.path.join(data_path, 'train_y.npy'))
    x_test = np.load(os.path.join(data_path, 'test_x.npy'))
    y_test = np.load(os.path.join(data_path, 'test_y.npy'))

    return x_train, y_train, x_test, y_test

def evaluate_checkpoint(dataset):
    """Evaluate a checkpoint on given dataset"""
    from CALANet_local.models_gts import HTAggNet

    checkpoint_path = os.path.join(ORIGINAL_CHECKPOINT_DIR, f"{dataset}.pt")
    if not os.path.exists(checkpoint_path):
        return None, f"Checkpoint not found: {checkpoint_path}"

    try:
        # Load data
        x_train, y_train, x_test, y_test = load_har_data(dataset)

        # Get dimensions
        channel = x_train.shape[2]
        seq_len = x_train.shape[1]
        n_classes = len(np.unique(y_train))

        print(f"  Data shape: {x_test.shape}, classes: {n_classes}")

        # Create model with same architecture
        # HTAggNet signature: (nc_input, n_classes, segment_size, L)
        L = 8  # Same as in run.py
        model = HTAggNet(channel, n_classes, seq_len, L).cuda()

        # Load checkpoint
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        # Prepare test data
        x_test_tensor = torch.FloatTensor(x_test).cuda()

        # Get predictions
        with torch.no_grad():
            y_pred = model(x_test_tensor).cpu().numpy()

        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_unary = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

        # Calculate F1-weighted
        f1_weighted = f1_score(y_test_unary, y_pred_classes, average='weighted')

        return f1_weighted * 100, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)

def main():
    print("="*80)
    print("EVALUATING ORIGINAL CHECKPOINTS FROM RTHAR.zip")
    print("="*80)
    print()

    print(f"{'Dataset':<20} {'Paper F1':<12} {'Original Ckpt':<15} {'Gap':<10} {'Status'}")
    print("-"*80)

    for dataset in HAR_DATASETS:
        paper_f1 = PAPER_HAR_F1["CALANet"].get(dataset, 0)

        print(f"Evaluating {dataset}...")
        f1, error = evaluate_checkpoint(dataset)

        if f1 is not None:
            gap = paper_f1 - f1
            status = "✓ MATCH" if abs(gap) < 1.0 else ("≈ CLOSE" if abs(gap) < 3.0 else "✗ DIFFERS")
            print(f"\r{dataset:<20} {paper_f1:<12.1f} {f1:<15.1f} {gap:<+10.1f} {status}")
        else:
            print(f"\r{dataset:<20} {paper_f1:<12.1f} {'ERROR':<15} {'-':<10} {error[:30]}")

    print("-"*80)
    print()
    print("If original checkpoints match paper values, we can use them directly!")
    print("="*80)

if __name__ == "__main__":
    main()
