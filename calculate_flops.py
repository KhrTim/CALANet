"""
Unified FLOPs calculation for all models
Calculates FLOPs and parameters for SAGOG, GTWIDL, MPTSNet, MSDL on all datasets
"""

import torch
import numpy as np
import sys
import os
from thop import profile, clever_format

# Add code paths
sys.path.insert(0, 'codes/CALANet_local')
sys.path.append('codes/SAGOG')
sys.path.append('codes/MPTSNet')
sys.path.append('codes/MSDL')

from utils import data_info

# Import models
from sagog_model import SAGoG
from model.MPTSNet import Model as MPTSNet
from msdl import MSDL

# Dataset information
HAR_DATASETS = ["UCI_HAR", "DSADS", "OPPORTUNITY", "KU-HAR", "PAMAP2", "REALDISP"]

TSC_DATASETS = {
    "AtrialFibrillation": {"channels": 2, "length": 640, "classes": 3},
    "MotorImagery": {"channels": 64, "length": 3000, "classes": 2},
    "Heartbeat": {"channels": 61, "length": 405, "classes": 2},
    "PhonemeSpectra": {"channels": 11, "length": 217, "classes": 39},
    "LSST": {"channels": 6, "length": 36, "classes": 14},
    "PEMS-SF": {"channels": 963, "length": 144, "classes": 7}
}

def calculate_flops_sagog(input_nc, segment_size, class_num):
    """Calculate FLOPs for SAGOG model"""
    # Adaptive configuration based on channels
    if input_nc >= 500:
        hidden_dim, graph_hidden_dim, num_graph_layers = 16, 32, 1
        num_windows = 3
    elif input_nc >= 50:
        hidden_dim, graph_hidden_dim, num_graph_layers = 32, 64, 1
        num_windows = 3
    else:
        hidden_dim, graph_hidden_dim, num_graph_layers = 64, 128, 2
        if segment_size >= 1000:
            num_windows = 10
        elif segment_size >= 500:
            num_windows = 8
        elif segment_size >= 200:
            num_windows = 5
        else:
            num_windows = 3

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
    )

    input_tensor = torch.rand(1, input_nc, segment_size)
    macs, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([macs*2, params], "%.3f")

    return flops, params

def calculate_flops_mptsnet(input_nc, segment_size, class_num):
    """Calculate FLOPs for MPTSNet model"""
    # Adaptive configuration
    if input_nc >= 500:
        embed_dim, embed_dim_t = 32, 128
    elif input_nc >= 100:
        embed_dim, embed_dim_t = 48, 192
    else:
        embed_dim = max(min(input_nc * 4, 256), 64)
        embed_dim_t = max(min(embed_dim * 4, 512), 256)

    # Default periods
    periods = [max(2, segment_size // 8), max(2, segment_size // 16), max(2, segment_size // 32)]
    periods = [p for p in periods if p > 1][:3]

    model = MPTSNet(
        periods=periods,
        flag=False,
        num_channels=input_nc,
        seq_length=segment_size,
        num_classes=class_num,
        embed_dim=embed_dim,
        embed_dim_t=embed_dim_t,
        num_heads=4,
        ff_dim=256,
        num_layers=1
    )

    input_tensor = torch.rand(1, input_nc, segment_size)
    macs, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([macs*2, params], "%.3f")

    return flops, params

def calculate_flops_msdl(input_nc, segment_size, class_num):
    """Calculate FLOPs for MSDL model"""
    # Adaptive configuration
    if input_nc >= 500:
        multiscale_channels, lstm_hidden = 32, 64
    elif input_nc >= 100:
        multiscale_channels, lstm_hidden = 48, 96
    else:
        multiscale_channels, lstm_hidden = 64, 128

    model = MSDL(
        input_channels=input_nc,
        num_classes=class_num,
        multiscale_channels=multiscale_channels,
        kernel_sizes=[3, 5, 7, 9],
        lstm_hidden=lstm_hidden,
        lstm_layers=2,
        dropout=0.5
    )

    input_tensor = torch.rand(1, input_nc, segment_size)
    macs, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([macs*2, params], "%.3f")

    return flops, params

def main():
    results = {}

    print("="*80)
    print("FLOPS CALCULATION FOR ALL MODELS")
    print("="*80)

    # HAR Datasets
    print("\n" + "="*80)
    print("HAR DATASETS")
    print("="*80)

    for dataset in HAR_DATASETS:
        print(f"\n{dataset}:")
        input_nc, segment_size, class_num = data_info(dataset)
        print(f"  Channels: {input_nc}, Length: {segment_size}, Classes: {class_num}")

        results[dataset] = {}

        try:
            flops, params = calculate_flops_sagog(input_nc, segment_size, class_num)
            results[dataset]['SAGOG'] = {'flops': flops, 'params': params}
            print(f"  SAGOG    - FLOPs: {flops:>10s}, Params: {params:>10s}")
        except Exception as e:
            print(f"  SAGOG    - Error: {e}")
            results[dataset]['SAGOG'] = {'flops': 'ERROR', 'params': 'ERROR'}

        try:
            flops, params = calculate_flops_mptsnet(input_nc, segment_size, class_num)
            results[dataset]['MPTSNet'] = {'flops': flops, 'params': params}
            print(f"  MPTSNet  - FLOPs: {flops:>10s}, Params: {params:>10s}")
        except Exception as e:
            print(f"  MPTSNet  - Error: {e}")
            results[dataset]['MPTSNet'] = {'flops': 'ERROR', 'params': 'ERROR'}

        try:
            flops, params = calculate_flops_msdl(input_nc, segment_size, class_num)
            results[dataset]['MSDL'] = {'flops': flops, 'params': params}
            print(f"  MSDL     - FLOPs: {flops:>10s}, Params: {params:>10s}")
        except Exception as e:
            print(f"  MSDL     - Error: {e}")
            results[dataset]['MSDL'] = {'flops': 'ERROR', 'params': 'ERROR'}

    # TSC Datasets
    print("\n" + "="*80)
    print("TSC DATASETS")
    print("="*80)

    for dataset, info in TSC_DATASETS.items():
        print(f"\n{dataset}:")
        input_nc = info['channels']
        segment_size = info['length']
        class_num = info['classes']
        print(f"  Channels: {input_nc}, Length: {segment_size}, Classes: {class_num}")

        results[dataset] = {}

        try:
            flops, params = calculate_flops_sagog(input_nc, segment_size, class_num)
            results[dataset]['SAGOG'] = {'flops': flops, 'params': params}
            print(f"  SAGOG    - FLOPs: {flops:>10s}, Params: {params:>10s}")
        except Exception as e:
            print(f"  SAGOG    - Error: {e}")
            results[dataset]['SAGOG'] = {'flops': 'ERROR', 'params': 'ERROR'}

        try:
            flops, params = calculate_flops_mptsnet(input_nc, segment_size, class_num)
            results[dataset]['MPTSNet'] = {'flops': flops, 'params': params}
            print(f"  MPTSNet  - FLOPs: {flops:>10s}, Params: {params:>10s}")
        except Exception as e:
            print(f"  MPTSNet  - Error: {e}")
            results[dataset]['MPTSNet'] = {'flops': 'ERROR', 'params': 'ERROR'}

        try:
            flops, params = calculate_flops_msdl(input_nc, segment_size, class_num)
            results[dataset]['MSDL'] = {'flops': flops, 'params': params}
            print(f"  MSDL     - FLOPs: {flops:>10s}, Params: {params:>10s}")
        except Exception as e:
            print(f"  MSDL     - Error: {e}")
            results[dataset]['MSDL'] = {'flops': 'ERROR', 'params': 'ERROR'}

    # Save results to file
    print("\n" + "="*80)
    print("Saving results to flops_results.txt")
    print("="*80)

    with open('flops_results.txt', 'w') as f:
        f.write("FLOPs and Parameters for All Models\n")
        f.write("="*80 + "\n\n")

        f.write("HAR DATASETS\n")
        f.write("="*80 + "\n")
        for dataset in HAR_DATASETS:
            f.write(f"\n{dataset}:\n")
            for model in ['SAGOG', 'MPTSNet', 'MSDL']:
                if model in results[dataset]:
                    f.write(f"  {model:10s} - FLOPs: {results[dataset][model]['flops']:>10s}, Params: {results[dataset][model]['params']:>10s}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("TSC DATASETS\n")
        f.write("="*80 + "\n")
        for dataset in TSC_DATASETS.keys():
            f.write(f"\n{dataset}:\n")
            for model in ['SAGOG', 'MPTSNet', 'MSDL']:
                if model in results[dataset]:
                    f.write(f"  {model:10s} - FLOPs: {results[dataset][model]['flops']:>10s}, Params: {results[dataset][model]['params']:>10s}\n")

    print("\nDone! Results saved to flops_results.txt")

if __name__ == "__main__":
    main()
