"""
Shared Metrics Collection Framework
====================================
Unified metrics collection for all models to ensure consistent evaluation.

Metrics Collected:
- Effectiveness: Accuracy, Precision, Recall, F1-scores, Confusion Matrix
- Efficiency: Training time, Inference time, Throughput, GPU memory, Parameters, FLOPs
"""

import time
import json
import os
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from datetime import datetime
from contextlib import contextmanager


class MetricsCollector:
    """
    Collects and manages all metrics for a single experiment run.

    Usage:
        collector = MetricsCollector(model_name='SAGOG', dataset='UCI_HAR',
                                     task_type='HAR', save_dir='results')

        # During training
        with collector.track_training_epoch():
            # ... training code ...
            pass

        # During inference
        with collector.track_inference():
            y_pred = model(x_test)

        # Compute final metrics
        collector.compute_classification_metrics(y_true, y_pred)
        collector.compute_model_complexity(model, input_shape)

        # Save all metrics
        collector.save_metrics()
    """

    def __init__(self, model_name, dataset, task_type, save_dir='results', run_id=None):
        """
        Args:
            model_name: Name of the model (e.g., 'SAGOG', 'GTWIDL')
            dataset: Dataset name (e.g., 'UCI_HAR', 'Heartbeat')
            task_type: Either 'HAR' or 'TSC'
            save_dir: Directory to save results
            run_id: Optional run identifier (defaults to timestamp)
        """
        self.model_name = model_name
        self.dataset = dataset
        self.task_type = task_type
        self.save_dir = save_dir
        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')

        # Initialize metrics storage
        self.metrics = {
            'model': model_name,
            'dataset': dataset,
            'task_type': task_type,
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'effectiveness': {},
            'efficiency': {},
            'training_history': {
                'epoch_times': [],
                'train_losses': [],
                'val_losses': [],
                'train_accs': [],
                'val_accs': []
            }
        }

        # Timing trackers
        self._training_start = None
        self._inference_start = None
        self._epoch_start = None

        # GPU memory tracking
        self._peak_memory_allocated = 0
        self._peak_memory_reserved = 0

    @contextmanager
    def track_training_epoch(self):
        """Context manager to track a single training epoch."""
        self._epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            yield
        finally:
            epoch_time = time.time() - self._epoch_start
            self.metrics['training_history']['epoch_times'].append(epoch_time)

            if torch.cuda.is_available():
                self._peak_memory_allocated = max(
                    self._peak_memory_allocated,
                    torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
                )
                self._peak_memory_reserved = max(
                    self._peak_memory_reserved,
                    torch.cuda.max_memory_reserved() / 1024**3
                )

    def record_epoch_metrics(self, train_loss=None, val_loss=None,
                           train_acc=None, val_acc=None):
        """Record metrics for a single epoch."""
        if train_loss is not None:
            self.metrics['training_history']['train_losses'].append(float(train_loss))
        if val_loss is not None:
            self.metrics['training_history']['val_losses'].append(float(val_loss))
        if train_acc is not None:
            self.metrics['training_history']['train_accs'].append(float(train_acc))
        if val_acc is not None:
            self.metrics['training_history']['val_accs'].append(float(val_acc))

    @contextmanager
    def track_training(self):
        """Context manager to track entire training process."""
        self._training_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            yield
        finally:
            total_time = time.time() - self._training_start
            self.metrics['efficiency']['training_time_seconds'] = total_time
            self.metrics['efficiency']['training_time_minutes'] = total_time / 60

            if torch.cuda.is_available():
                self.metrics['efficiency']['peak_memory_allocated_gb'] = \
                    torch.cuda.max_memory_allocated() / 1024**3
                self.metrics['efficiency']['peak_memory_reserved_gb'] = \
                    torch.cuda.max_memory_reserved() / 1024**3

    @contextmanager
    def track_inference(self):
        """Context manager to track inference time."""
        self._inference_start = time.time()

        try:
            yield
        finally:
            inference_time = time.time() - self._inference_start
            self.metrics['efficiency']['inference_time_seconds'] = inference_time

    def compute_throughput(self, num_samples, phase='inference'):
        """
        Compute throughput (samples/second).

        Args:
            num_samples: Number of samples processed
            phase: Either 'training' or 'inference'
        """
        if phase == 'training':
            time_key = 'training_time_seconds'
        else:
            time_key = 'inference_time_seconds'

        if time_key in self.metrics['efficiency']:
            time_taken = self.metrics['efficiency'][time_key]
            throughput = num_samples / time_taken if time_taken > 0 else 0
            self.metrics['efficiency'][f'{phase}_throughput_samples_per_sec'] = throughput

    def compute_classification_metrics(self, y_true, y_pred, class_names=None):
        """
        Compute all classification effectiveness metrics.

        Args:
            y_true: Ground truth labels (numpy array or torch tensor)
            y_pred: Predicted labels (numpy array or torch tensor)
            class_names: Optional list of class names
        """
        # Convert to numpy if needed
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()

        # Handle predictions that are probabilities/logits
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        # Compute basic metrics
        self.metrics['effectiveness']['accuracy'] = float(accuracy_score(y_true, y_pred))

        # Compute precision, recall, f1 with different averaging strategies
        for avg in ['macro', 'weighted', 'micro']:
            self.metrics['effectiveness'][f'precision_{avg}'] = float(
                precision_score(y_true, y_pred, average=avg, zero_division=0)
            )
            self.metrics['effectiveness'][f'recall_{avg}'] = float(
                recall_score(y_true, y_pred, average=avg, zero_division=0)
            )
            self.metrics['effectiveness'][f'f1_{avg}'] = float(
                f1_score(y_true, y_pred, average=avg, zero_division=0)
            )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.metrics['effectiveness']['confusion_matrix'] = cm.tolist()

        # Detailed per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        self.metrics['effectiveness']['per_class_metrics'] = {}

        for key, value in report.items():
            if key.isdigit() or key in ['macro avg', 'weighted avg']:
                self.metrics['effectiveness']['per_class_metrics'][key] = value

    def compute_model_complexity(self, model, input_shape=None, device='cuda'):
        """
        Compute model complexity metrics (parameters, FLOPs).

        Args:
            model: PyTorch model
            input_shape: Tuple representing input shape (batch_size, ...)
                        If None, only parameters are counted
            device: Device to use for computation
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.metrics['efficiency']['total_parameters'] = int(total_params)
        self.metrics['efficiency']['trainable_parameters'] = int(trainable_params)
        self.metrics['efficiency']['parameters_millions'] = total_params / 1e6

        # Compute FLOPs if input shape provided
        if input_shape is not None:
            try:
                from thop import profile, clever_format

                # Create dummy input
                if isinstance(input_shape, (list, tuple)):
                    dummy_input = torch.randn(*input_shape).to(device)
                else:
                    dummy_input = input_shape.to(device)

                # Profile model
                flops, params = profile(model, inputs=(dummy_input,), verbose=False)
                flops_formatted, params_formatted = clever_format([flops, params], "%.2f")

                self.metrics['efficiency']['flops'] = int(flops)
                self.metrics['efficiency']['flops_formatted'] = flops_formatted

            except ImportError:
                print("Warning: 'thop' package not found. FLOPs will not be computed.")
            except Exception as e:
                print(f"Warning: Could not compute FLOPs: {e}")

    def add_custom_metric(self, category, name, value):
        """
        Add a custom metric.

        Args:
            category: Either 'effectiveness' or 'efficiency'
            name: Metric name
            value: Metric value
        """
        if category not in ['effectiveness', 'efficiency']:
            raise ValueError("category must be 'effectiveness' or 'efficiency'")

        self.metrics[category][name] = value

    def save_metrics(self, filename=None):
        """
        Save all collected metrics to a JSON file.

        Args:
            filename: Optional custom filename. If None, uses default naming convention.
        """
        # Compute summary statistics
        if self.metrics['training_history']['epoch_times']:
            self.metrics['efficiency']['avg_epoch_time_seconds'] = \
                np.mean(self.metrics['training_history']['epoch_times'])
            self.metrics['efficiency']['total_epochs'] = \
                len(self.metrics['training_history']['epoch_times'])

        # Create save directory
        save_path = os.path.join(self.save_dir, self.model_name)
        os.makedirs(save_path, exist_ok=True)

        # Generate filename
        if filename is None:
            filename = f"{self.dataset}_metrics.json"

        filepath = os.path.join(save_path, filename)

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"\nMetrics saved to: {filepath}")

        # Also save a human-readable summary
        summary_file = os.path.join(save_path, f"{self.dataset}_summary.txt")
        self._save_summary(summary_file)

        return filepath

    def _save_summary(self, filepath):
        """Save a human-readable summary of metrics."""
        with open(filepath, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"{self.model_name} Results on {self.dataset}\n")
            f.write(f"{'='*70}\n\n")

            # Effectiveness metrics
            f.write("EFFECTIVENESS METRICS:\n")
            f.write("-" * 70 + "\n")
            eff = self.metrics['effectiveness']
            if 'accuracy' in eff:
                f.write(f"Accuracy:           {eff['accuracy']:.4f}\n")
            if 'precision_weighted' in eff:
                f.write(f"Precision (Weighted): {eff['precision_weighted']:.4f}\n")
            if 'recall_weighted' in eff:
                f.write(f"Recall (Weighted):    {eff['recall_weighted']:.4f}\n")
            if 'f1_weighted' in eff:
                f.write(f"F1-Score (Weighted):  {eff['f1_weighted']:.4f}\n")
            if 'f1_macro' in eff:
                f.write(f"F1-Score (Macro):     {eff['f1_macro']:.4f}\n")

            # Efficiency metrics
            f.write("\n" + "="*70 + "\n")
            f.write("EFFICIENCY METRICS:\n")
            f.write("-" * 70 + "\n")
            eff_metrics = self.metrics['efficiency']
            if 'training_time_minutes' in eff_metrics:
                f.write(f"Training Time:        {eff_metrics['training_time_minutes']:.2f} minutes\n")
            if 'inference_time_seconds' in eff_metrics:
                f.write(f"Inference Time:       {eff_metrics['inference_time_seconds']:.4f} seconds\n")
            if 'inference_throughput_samples_per_sec' in eff_metrics:
                f.write(f"Inference Throughput: {eff_metrics['inference_throughput_samples_per_sec']:.2f} samples/sec\n")
            if 'peak_memory_allocated_gb' in eff_metrics:
                f.write(f"Peak GPU Memory:      {eff_metrics['peak_memory_allocated_gb']:.2f} GB\n")
            if 'parameters_millions' in eff_metrics:
                f.write(f"Parameters:           {eff_metrics['parameters_millions']:.2f}M\n")
            if 'flops_formatted' in eff_metrics:
                f.write(f"FLOPs:                {eff_metrics['flops_formatted']}\n")

            # Confusion Matrix
            if 'confusion_matrix' in self.metrics['effectiveness']:
                f.write("\n" + "="*70 + "\n")
                f.write("CONFUSION MATRIX:\n")
                f.write("-" * 70 + "\n")
                cm = np.array(self.metrics['effectiveness']['confusion_matrix'])
                f.write(str(cm) + "\n")

            f.write("\n" + "="*70 + "\n")

    def print_summary(self):
        """Print a summary of collected metrics to console."""
        print(f"\n{'='*70}")
        print(f"{self.model_name} - {self.dataset} ({self.task_type})")
        print(f"{'='*70}")

        if self.metrics['effectiveness']:
            print("\nEffectiveness Metrics:")
            for key, value in self.metrics['effectiveness'].items():
                if key not in ['confusion_matrix', 'per_class_metrics']:
                    if isinstance(value, float):
                        print(f"  {key:25s}: {value:.4f}")

        if self.metrics['efficiency']:
            print("\nEfficiency Metrics:")
            for key, value in self.metrics['efficiency'].items():
                if isinstance(value, (int, float)):
                    if 'time' in key.lower():
                        print(f"  {key:35s}: {value:.4f}")
                    else:
                        print(f"  {key:35s}: {value}")


def load_metrics(filepath):
    """
    Load metrics from a JSON file.

    Args:
        filepath: Path to the metrics JSON file

    Returns:
        Dictionary containing the metrics
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_metrics(metrics_list, metric_names=None):
    """
    Compare metrics across multiple runs or models.

    Args:
        metrics_list: List of metric dictionaries (or file paths)
        metric_names: Optional list of specific metrics to compare

    Returns:
        DataFrame with comparison results
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for metric comparison")
        return None

    # Load metrics if filepaths provided
    loaded_metrics = []
    for m in metrics_list:
        if isinstance(m, str):
            loaded_metrics.append(load_metrics(m))
        else:
            loaded_metrics.append(m)

    # Extract comparison data
    comparison_data = []
    for m in loaded_metrics:
        row = {
            'model': m.get('model', 'Unknown'),
            'dataset': m.get('dataset', 'Unknown'),
        }

        # Add effectiveness metrics
        for key, value in m.get('effectiveness', {}).items():
            if not isinstance(value, (list, dict)):
                row[f'eff_{key}'] = value

        # Add efficiency metrics
        for key, value in m.get('efficiency', {}).items():
            if not isinstance(value, (list, dict, str)):
                row[f'effi_{key}'] = value

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Filter to specific metrics if requested
    if metric_names:
        cols_to_keep = ['model', 'dataset'] + [col for col in df.columns if any(m in col for m in metric_names)]
        df = df[cols_to_keep]

    return df
