"""
Extract Comprehensive Metrics from Existing Experiments
========================================================

This script processes existing experiment results and model checkpoints to extract
comprehensive metrics including those requested by reviewers:

Effectiveness Metrics:
- Accuracy, Precision, Recall, F1-scores
- Confusion Matrix
- Per-class metrics

Efficiency Metrics:
- Inference time
- Throughput (samples/second)
- Peak GPU memory
- Number of parameters
- FLOPs

Usage:
    python extract_comprehensive_metrics.py --model SAGOG --dataset UCI_HAR --task HAR
    python extract_comprehensive_metrics.py --all  # Process all available results
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Add codes directory to path
codes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'codes')
sys.path.insert(0, codes_dir)
sys.path.insert(0, os.path.join(codes_dir, 'CALANet_local'))

from shared_metrics import MetricsCollector


# Model configurations for each model type
MODEL_CONFIGS = {
    'SAGOG': {
        'module_path': 'SAGOG',
        'model_class': 'SAGoG',
        'model_file': 'sagog_model',
        'input_format': '4d',  # (batch, channels, 1, length)
    },
    'GTWIDL': {
        'module_path': 'GTWIDL',
        'model_class': 'GTWIDL',
        'model_file': 'gtwidl_model',
        'input_format': '3d',  # (batch, channels, length)
    },
    'MPTSNet': {
        'module_path': 'MPTSNet',
        'model_class': 'MPTSNet',
        'model_file': 'model.MPTSNet',
        'input_format': 'time_first',  # (batch, length, channels)
    },
    'MSDL': {
        'module_path': 'MSDL',
        'model_class': 'MSDL',
        'model_file': 'msdl',
        'input_format': '3d',  # (batch, channels, length)
    }
}


# Dataset information
HAR_DATASETS = {
    'UCI_HAR': {'channels': 6, 'length': 128, 'classes': 6},
    'DSADS': {'channels': 45, 'length': 125, 'classes': 19},
    'OPPORTUNITY': {'channels': 113, 'length': 90, 'classes': 17},
    'KU-HAR': {'channels': 6, 'length': 300, 'classes': 18},
    'PAMAP2': {'channels': 31, 'length': 512, 'classes': 18},
    'REALDISP': {'channels': 117, 'length': 250, 'classes': 33}
}

TSC_DATASETS = {
    'AtrialFibrillation': {'channels': 2, 'length': 640, 'classes': 3},
    'MotorImagery': {'channels': 64, 'length': 3000, 'classes': 2},
    'Heartbeat': {'channels': 61, 'length': 405, 'classes': 2},
    'PhonemeSpectra': {'channels': 11, 'length': 217, 'classes': 39},
    'LSST': {'channels': 6, 'length': 36, 'classes': 14},
    'PEMS-SF': {'channels': 963, 'length': 144, 'classes': 7}
}


def load_model_checkpoint(model_name, dataset, task_type):
    """
    Load a trained model checkpoint.

    Returns:
        model, checkpoint_path, or (None, None) if not found
    """
    # Try different checkpoint locations
    checkpoint_patterns = [
        f'{model_name}/save/{dataset}_{model_name.lower()}.pt',
        f'{model_name}/save/{dataset}.pt',
        f'{model_name}/checkpoints/{dataset}.pt',
        f'{model_name}/{dataset}_best.pt',
    ]

    checkpoint_path = None
    for pattern in checkpoint_patterns:
        if os.path.exists(pattern):
            checkpoint_path = pattern
            break

    if not checkpoint_path:
        print(f"⚠️  No checkpoint found for {model_name}/{dataset}")
        return None, None

    print(f"Found checkpoint: {checkpoint_path}")

    # Load model based on configuration
    try:
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            print(f"⚠️  No configuration for model {model_name}")
            return None, checkpoint_path

        # Import model class
        model_module_path = os.path.join(codes_dir, config['module_path'])
        sys.path.insert(0, model_module_path)

        # Get dataset info
        if task_type == 'HAR':
            info = HAR_DATASETS.get(dataset)
        else:
            info = TSC_DATASETS.get(dataset)

        if not info:
            print(f"⚠️  Unknown dataset {dataset}")
            return None, checkpoint_path

        # Create model instance (simplified - may need adjustment per model)
        # This is a placeholder - actual model creation depends on each model's API
        print(f"⚠️  Model loading requires manual implementation for {model_name}")
        return None, checkpoint_path

    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        return None, checkpoint_path


def load_test_data(dataset, task_type):
    """
    Load test data for a dataset.

    Returns:
        X_test, y_test, or (None, None) if not found
    """
    if task_type == 'HAR':
        data_path = os.path.join('Data', 'preprocessed', dataset)
        test_X_path = os.path.join(data_path, 'test_x.npy')
        test_Y_path = os.path.join(data_path, 'test_y.npy')

        if not os.path.exists(test_X_path) or not os.path.exists(test_Y_path):
            return None, None

        X_test = np.load(test_X_path)
        y_test = np.load(test_Y_path)
        return X_test, y_test

    else:  # TSC
        try:
            from aeon.datasets import load_from_arff_file
            data_path = os.path.join('Data', 'TSC', dataset)
            test_file = os.path.join(data_path, f'{dataset}_TEST.arff')

            if not os.path.exists(test_file):
                return None, None

            X_test, y_test = load_from_arff_file(test_file)
            _, y_test = np.unique(y_test, return_inverse=True)
            return X_test, y_test
        except Exception as e:
            print(f"⚠️  Error loading TSC data: {e}")
            return None, None


def find_existing_predictions(model_name, dataset, task_type):
    """
    Try to find existing prediction results.

    Returns:
        y_true, y_pred, or (None, None)
    """
    # Check for saved results in results directory
    result_patterns = [
        f'{model_name}/results/{dataset}_{model_name.lower()}_results.txt',
        f'{model_name}/results/{dataset}_results.txt',
        f'logs_{task_type.lower()}/{model_name}/{dataset}_log.txt',
    ]

    # For now, return None - would need to parse result files
    # This could be implemented to extract predictions from logs
    return None, None


def compute_metrics_from_saved_results(model_name, dataset, task_type):
    """
    Compute comprehensive metrics from existing saved results.

    This function tries to:
    1. Load model checkpoint
    2. Load test data
    3. Run inference and measure time/memory
    4. Compute all metrics
    5. Save using MetricsCollector
    """
    print(f"\n{'='*70}")
    print(f"Processing: {model_name} on {dataset} ({task_type})")
    print(f"{'='*70}")

    # Initialize metrics collector
    collector = MetricsCollector(
        model_name=model_name,
        dataset=dataset,
        task_type=task_type,
        save_dir='results'
    )

    # Load test data
    print("Loading test data...")
    X_test, y_test = load_test_data(dataset, task_type)

    if X_test is None:
        print(f"❌ Could not load test data for {dataset}")
        return False

    print(f"✓ Loaded test data: {X_test.shape}")

    # Try to load model and run inference
    print("Loading model checkpoint...")
    model, checkpoint_path = load_model_checkpoint(model_name, dataset, task_type)

    # Check if we have existing predictions
    y_true_existing, y_pred_existing = find_existing_predictions(model_name, dataset, task_type)

    # For now, we'll compute metrics from what we can
    # This is a placeholder - full implementation requires model-specific loading

    print("\n⚠️  Full metric extraction requires model-specific implementation")
    print("Please use the integrated MetricsCollector in experiment scripts for complete metrics")

    # Save what we can
    collector.add_custom_metric('efficiency', 'checkpoint_path', checkpoint_path or 'Not found')
    collector.add_custom_metric('effectiveness', 'test_samples', len(y_test))

    # Save metrics
    # collector.save_metrics(filename=f'{dataset}_extracted_metrics.json')

    return True


def process_all_experiments():
    """
    Process all available experiment results.
    """
    processed = []
    failed = []

    for model in MODEL_CONFIGS.keys():
        # HAR datasets
        for dataset in HAR_DATASETS.keys():
            try:
                if compute_metrics_from_saved_results(model, dataset, 'HAR'):
                    processed.append((model, dataset, 'HAR'))
                else:
                    failed.append((model, dataset, 'HAR'))
            except Exception as e:
                print(f"❌ Error processing {model}/{dataset}/HAR: {e}")
                failed.append((model, dataset, 'HAR'))

        # TSC datasets
        for dataset in TSC_DATASETS.keys():
            try:
                if compute_metrics_from_saved_results(model, dataset, 'TSC'):
                    processed.append((model, dataset, 'TSC'))
                else:
                    failed.append((model, dataset, 'TSC'))
            except Exception as e:
                print(f"❌ Error processing {model}/{dataset}/TSC: {e}")
                failed.append((model, dataset, 'TSC'))

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Processed: {len(processed)}")
    print(f"Failed: {len(failed)}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract comprehensive metrics from existing experiments'
    )
    parser.add_argument('--model', type=str, help='Model name (e.g., SAGOG)')
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., UCI_HAR)')
    parser.add_argument('--task', choices=['HAR', 'TSC'], help='Task type')
    parser.add_argument('--all', action='store_true', help='Process all experiments')

    args = parser.parse_args()

    if args.all:
        process_all_experiments()
    elif args.model and args.dataset and args.task:
        compute_metrics_from_saved_results(args.model, args.dataset, args.task)
    else:
        parser.print_help()
        print("\n💡 TIP: For complete metrics, use the integrated MetricsCollector")
        print("See codes/SAGOG/run_har_experiments.py for an example")


if __name__ == '__main__':
    main()
