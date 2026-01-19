#!/usr/bin/env python3
"""
Test all original checkpoints from RTHAR.zip
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'codes'))

# Models with checkpoints in RTHAR.zip
MODELS_WITH_CHECKPOINTS = {
    'resnet': {
        'module': 'resnet.model',
        'class': 'resnet',
        'datasets': ['UCI_HAR', 'OPPORTUNITY', 'PAMAP2'],
        'checkpoint_dir': '/tmp/original_checkpoints/codes/resnet/save',
    },
    'Bi-GRU-I': {
        'module': 'Bi-GRU-I.models',
        'class': 'BiGRU',
        'datasets': ['UCI_HAR', 'DSADS', 'OPPORTUNITY', 'KU-HAR', 'PAMAP2', 'REALDISP'],
        'checkpoint_dir': '/tmp/original_checkpoints/codes/Bi-GRU-I/save',
    },
    'DSN-master': {
        'module': 'DSN-master.model.SCNN',
        'class': 'SCNN',
        'datasets': ['UCI_HAR', 'DSADS', 'OPPORTUNITY', 'KU-HAR', 'PAMAP2', 'REALDISP'],
        'checkpoint_dir': '/tmp/original_checkpoints/codes/DSN-master/save',
    },
}

def load_data(dataset):
    """Load dataset"""
    data_path = f"Data/preprocessed/{dataset}"
    x_test = np.load(f"{data_path}/test_x.npy")
    y_test = np.load(f"{data_path}/test_y.npy")
    return x_test, y_test

def get_data_info(dataset):
    """Get dataset info"""
    info = {
        'UCI_HAR': (6, 128, 6),
        'DSADS': (45, 125, 19),
        'OPPORTUNITY': (113, 90, 17),
        'KU-HAR': (6, 300, 18),
        'PAMAP2': (31, 512, 18),
        'REALDISP': (117, 250, 33),
    }
    return info.get(dataset, (None, None, None))

def test_model(model_name, model_info):
    print(f"\n{'='*70}")
    print(f"Testing {model_name}")
    print(f"{'='*70}")

    # Extract checkpoints if not already done
    ckpt_dir = model_info['checkpoint_dir']
    if not os.path.exists(ckpt_dir):
        import subprocess
        subprocess.run(['unzip', '-o', os.path.expanduser('~/RTHAR.zip'),
                       f"codes/{model_name}/save/*.pt", '-d', '/tmp/original_checkpoints'],
                      capture_output=True)

    results = []
    for dataset in model_info['datasets']:
        ckpt_path = f"{ckpt_dir}/{dataset}.pt"
        if not os.path.exists(ckpt_path):
            print(f"  {dataset}: checkpoint not found")
            continue

        try:
            # Load data
            x_test, y_test = load_data(dataset)
            y_test_labels = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)

            # Load checkpoint to check architecture
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            # Check first layer shape to understand architecture
            first_key = list(ckpt.keys())[0]
            first_shape = ckpt[first_key].shape

            print(f"  {dataset}: ckpt first layer {first_key} = {first_shape}")
            results.append((dataset, "loaded", first_shape))

        except Exception as e:
            print(f"  {dataset}: ERROR - {str(e)[:50]}")
            results.append((dataset, "error", str(e)[:50]))

    return results

def main():
    print("="*70)
    print("TESTING ORIGINAL CHECKPOINTS FROM RTHAR.zip")
    print("="*70)

    # Extract all checkpoints first
    print("\nExtracting checkpoints...")
    import subprocess
    for model in ['resnet', 'Bi-GRU-I', 'DSN-master', 'RevTransformerAttentionHAR']:
        subprocess.run(['unzip', '-o', '-q', os.path.expanduser('~/RTHAR.zip'),
                       f"codes/{model}/save/*.pt", '-d', '/tmp/original_checkpoints'],
                      capture_output=True)

    # Test each model
    for model_name, model_info in MODELS_WITH_CHECKPOINTS.items():
        test_model(model_name, model_info)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Models WITH original checkpoints in RTHAR.zip:
  - Bi-GRU-I: 7 checkpoints (HAR only)
  - DSN-master: 13 checkpoints (HAR + TSC)
  - FCN_TSC: 5 checkpoints
  - resnet: 5 checkpoints (HAR only)
  - RevTransformerAttentionHAR: 7 checkpoints
  - IF-ConvTransformer: 2 checkpoints

Models WITHOUT original checkpoints (trained from scratch):
  - CALANet (architecture mismatch with saved checkpoints)
  - millet
  - InceptionTime
  - MPTSNet
  - MSDL
  - RepHAR
  - SAGOG
  - DeepConvLSTM

The gap between paper results and our results likely comes from:
1. Different random seeds
2. Different hyperparameters
3. Architecture changes (as seen with CALANet)
4. Training from scratch vs using pre-trained checkpoints
""")

if __name__ == "__main__":
    main()
