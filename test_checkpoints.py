#!/usr/bin/env python3
"""
Quick test to verify checkpoint compatibility
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'codes'))
from CALANet_local.models_gts import HTAggNet
from sklearn.metrics import f1_score

# Paths
ORIGINAL_CKPT = "/tmp/original_checkpoints/codes/CALANet_local/save/with_gts/UCI_HAR.pt"
CURRENT_CKPT = "HT-AggNet_v2/save/with_gts/UCI_HAR.pt"

def load_data():
    data_path = "Data/preprocessed/UCI_HAR"
    x_test = np.load(f"{data_path}/test_x.npy")
    y_test = np.load(f"{data_path}/test_y.npy")
    return x_test, y_test

def test_checkpoint_modified_model(ckpt_path, name):
    """Test with modified model architecture (n_groups=4)"""
    print(f"\n{'='*60}")
    print(f"Testing with MODIFIED model (n_groups=4): {name}")
    print(f"Path: {ckpt_path}")
    print(f"{'='*60}")

    if not os.path.exists(ckpt_path):
        print("NOT FOUND")
        return

    # Load data
    x_test, y_test = load_data()

    # Check checkpoint info
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    channels = x_test.shape[1]  # 6
    seq_len = x_test.shape[2]   # 128
    y_test_labels = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)
    n_classes = len(np.unique(y_test_labels))  # 6
    L = 8

    # Import and modify the model
    import codes.CALANet_local.models_gts as models_gts

    # Monkey-patch TAggBlock to use n_groups=4
    original_TAggBlock_init = models_gts.TAggBlock.__init__
    def patched_init(self, i_nc, o_nc, L, T, pool):
        import torch.nn as nn
        import torch.nn.functional as F
        super(models_gts.TAggBlock, self).__init__()
        self.L = L
        self.pool = pool
        # Use n_groups=4 instead of o_nc//4 (which would be 16)
        self.gconv = models_gts.GTSConvUnit(i_nc, o_nc, 4)  # Changed from o_nc//4
        self.tgconv = nn.Conv1d(o_nc, o_nc//L, kernel_size=T)

    models_gts.TAggBlock.__init__ = patched_init

    try:
        model = models_gts.HTAggNet(channels, n_classes, seq_len, L).cuda()
        model.load_state_dict(ckpt)
        print("✓ Checkpoint loaded successfully with modified model!")

        # Run inference
        model.eval()
        x_test_tensor = torch.FloatTensor(x_test).cuda()
        with torch.no_grad():
            y_pred = model(x_test_tensor).cpu().numpy()

        y_pred_classes = np.argmax(y_pred, axis=1)

        f1 = f1_score(y_test_labels, y_pred_classes, average='weighted') * 100
        print(f"✓ F1-weighted: {f1:.1f}%")

    except Exception as e:
        print(f"✗ Error: {e}")

    finally:
        # Restore original
        models_gts.TAggBlock.__init__ = original_TAggBlock_init

def test_checkpoint(ckpt_path, name):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Path: {ckpt_path}")
    print(f"{'='*60}")

    if not os.path.exists(ckpt_path):
        print("NOT FOUND")
        return

    # Load data
    x_test, y_test = load_data()
    print(f"Data shape: {x_test.shape}")

    # Check checkpoint info
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"Checkpoint keys: {len(ckpt)} layers")

    # Print first few layer shapes
    for i, (k, v) in enumerate(ckpt.items()):
        if i < 5:
            print(f"  {k}: {v.shape}")
        elif i == 5:
            print("  ...")

    # Try to create model
    channels = x_test.shape[1]  # 6
    seq_len = x_test.shape[2]   # 128
    y_test_labels = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)
    n_classes = len(np.unique(y_test_labels))  # 6
    L = 8

    print(f"\nModel config: channels={channels}, seq_len={seq_len}, n_classes={n_classes}, L={L}")

    try:
        model = HTAggNet(channels, n_classes, seq_len, L).cuda()
        print(f"Model created successfully")

        # Try loading checkpoint
        model.load_state_dict(ckpt)
        print("✓ Checkpoint loaded successfully!")

        # Run inference
        model.eval()
        x_test_tensor = torch.FloatTensor(x_test).cuda()
        with torch.no_grad():
            y_pred = model(x_test_tensor).cpu().numpy()

        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)

        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted') * 100
        print(f"✓ F1-weighted: {f1:.1f}%")

    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_checkpoint(CURRENT_CKPT, "Current (HT-AggNet_v2)")
    test_checkpoint(ORIGINAL_CKPT, "Original from RTHAR.zip (CALANet_local)")
    test_checkpoint_modified_model(ORIGINAL_CKPT, "Original with n_groups=4")
