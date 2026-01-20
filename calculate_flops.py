#!/usr/bin/env python3
"""Calculate FLOPs for all models on all datasets"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
from thop import profile, clever_format

# Add paths
codes_dir = os.path.join(os.path.abspath('.'), 'codes')
sys.path.insert(0, os.path.join(codes_dir, 'CALANet_local'))

from utils import data_info

MODELS = ["SAGOG", "GTWIDL", "MPTSNet", "MSDL"]
HAR_DATASETS = ["UCI_HAR", "DSADS", "OPPORTUNITY", "KU-HAR", "PAMAP2", "REALDISP"]
TSC_DATASETS = ["AtrialFibrillation", "MotorImagery", "Heartbeat", "PhonemeSpectra", "LSST", "PEMS-SF"]

# TSC dataset info
TSC_INFO = {
    "AtrialFibrillation": {"channels": 2, "length": 640, "classes": 3},
    "MotorImagery": {"channels": 64, "length": 3000, "classes": 2},
    "Heartbeat": {"channels": 61, "length": 405, "classes": 2},
    "PhonemeSpectra": {"channels": 11, "length": 217, "classes": 39},
    "LSST": {"channels": 6, "length": 36, "classes": 14},
    "PEMS-SF": {"channels": 963, "length": 144, "classes": 7}
}

def get_sagog_flops(input_nc, segment_size, class_num):
    """Calculate FLOPs for SAGoG model"""
    sys.path.insert(0, os.path.join(codes_dir, 'SAGOG'))
    from sagog_model import SAGoG

    # Use adaptive config like in experiments
    if input_nc > 50:
        hidden_dim = 64
        num_window = 2
    elif input_nc > 20:
        hidden_dim = 128
        num_window = 3
    else:
        hidden_dim = 256
        num_window = 4

    model = SAGoG(
        input_nc=input_nc,
        input_height=1,
        input_width=segment_size,
        class_num=class_num,
        hidden_dim=hidden_dim,
        num_window=num_window
    )

    input_tensor = torch.randn(1, input_nc, 1, segment_size)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    return flops, params

def get_mptsnet_flops(input_nc, segment_size, class_num):
    """Calculate FLOPs for MPTSNet model"""
    sys.path.insert(0, os.path.join(codes_dir, 'MPTSNet'))
    sys.path.insert(0, os.path.join(codes_dir, 'MPTSNet', 'model'))
    from model.MPTSNet import MPTSNet

    # Adaptive config from experiments
    if input_nc > 50:
        d_model = 64
        n_heads = 4
    else:
        d_model = 128
        n_heads = 8

    model = MPTSNet(
        input_nc=input_nc,
        seq_len=segment_size,
        num_classes=class_num,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=256,
        num_layers=2
    )

    input_tensor = torch.randn(1, segment_size, input_nc)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    return flops, params

def get_msdl_flops(input_nc, segment_size, class_num):
    """Calculate FLOPs for MSDL model"""
    sys.path.insert(0, os.path.join(codes_dir, 'MSDL'))
    from msdl import MSDL

    model = MSDL(
        input_channels=input_nc,
        seq_length=segment_size,
        num_classes=class_num,
        hidden_dim=128,
        num_layers=2,
        kernel_sizes=[3, 5, 7]
    )

    input_tensor = torch.randn(1, input_nc, segment_size)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    return flops, params

def get_gtwidl_flops(input_nc, segment_size, class_num):
    """
    Estimate FLOPs for GTWIDL
    GTWIDL is dictionary-based, so FLOPs are mainly in:
    1. Transform (sparse coding) - iterative optimization
    2. SVM inference - not counted as it's sklearn
    """
    # GTWIDL doesn't have a standard forward pass
    # Estimate based on transform operations
    n_atoms = 3 if input_nc <= 50 else 4
    max_iter = 50  # transform iterations
    n_samples = 1

    # Main cost: soft thresholding iterations
    # Each iteration: matrix multiply (L*C) x n_atoms + soft threshold
    atom_size = segment_size * input_nc  # flattened size
    flops_per_iter = atom_size * n_atoms * 2  # matmul
    flops_per_iter += n_atoms * 2  # soft threshold

    total_flops = flops_per_iter * max_iter

    # Dictionary atoms
    params = n_atoms * segment_size * input_nc

    return total_flops, params

# Calculate FLOPs for all models and datasets
all_flops = {
    'HAR': {},
    'TSC': {}
}

print("="*80)
print("CALCULATING FLOPs FOR ALL MODELS")
print("="*80)

print("\nHAR DATASETS:")
print("-"*80)
for dataset in HAR_DATASETS:
    print(f"\n{dataset}:")
    input_nc, segment_size, class_num = data_info(dataset)
    print(f"  Channels: {input_nc}, Length: {segment_size}, Classes: {class_num}")

    for model in MODELS:
        if model not in all_flops['HAR']:
            all_flops['HAR'][model] = {}

        try:
            if model == "SAGOG":
                flops, params = get_sagog_flops(input_nc, segment_size, class_num)
            elif model == "MPTSNet":
                flops, params = get_mptsnet_flops(input_nc, segment_size, class_num)
            elif model == "MSDL":
                flops, params = get_msdl_flops(input_nc, segment_size, class_num)
            elif model == "GTWIDL":
                flops, params = get_gtwidl_flops(input_nc, segment_size, class_num)

            all_flops['HAR'][model][dataset] = {
                'flops': int(flops),
                'params': int(params)
            }

            flops_str, params_str = clever_format([flops, params], "%.2f")
            print(f"  {model:10s}: {flops_str:>10s} FLOPs, {params_str:>10s} params")

        except Exception as e:
            print(f"  {model:10s}: ERROR - {e}")
            all_flops['HAR'][model][dataset] = {'flops': 0, 'params': 0}

print("\n\nTSC DATASETS:")
print("-"*80)
for dataset in TSC_DATASETS:
    print(f"\n{dataset}:")
    info = TSC_INFO[dataset]
    input_nc, segment_size, class_num = info['channels'], info['length'], info['classes']
    print(f"  Channels: {input_nc}, Length: {segment_size}, Classes: {class_num}")

    for model in MODELS:
        if model not in all_flops['TSC']:
            all_flops['TSC'][model] = {}

        try:
            if model == "SAGOG":
                flops, params = get_sagog_flops(input_nc, segment_size, class_num)
            elif model == "MPTSNet":
                flops, params = get_mptsnet_flops(input_nc, segment_size, class_num)
            elif model == "MSDL":
                flops, params = get_msdl_flops(input_nc, segment_size, class_num)
            elif model == "GTWIDL":
                flops, params = get_gtwidl_flops(input_nc, segment_size, class_num)

            all_flops['TSC'][model][dataset] = {
                'flops': int(flops),
                'params': int(params)
            }

            flops_str, params_str = clever_format([flops, params], "%.2f")
            print(f"  {model:10s}: {flops_str:>10s} FLOPs, {params_str:>10s} params")

        except Exception as e:
            print(f"  {model:10s}: ERROR - {e}")
            all_flops['TSC'][model][dataset] = {'flops': 0, 'params': 0}

# Save to JSON
with open('all_flops.json', 'w') as f:
    json.dump(all_flops, f, indent=2)

print("\n" + "="*80)
print("FLOPs saved to all_flops.json")
print("="*80)
