# Metrics Integration Guide

This guide shows how to integrate the comprehensive metrics collection framework into existing model training scripts.

## Quick Integration Steps

### 1. Import the MetricsCollector

Add this to your imports section:

```python
import importlib.util
import os
import sys

# Get paths
current_dir = os.path.dirname(os.path.abspath(__file__))
codes_dir = os.path.dirname(current_dir)

# Import shared metrics collector
spec = importlib.util.spec_from_file_location("shared_metrics", os.path.join(codes_dir, 'shared_metrics.py'))
shared_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_metrics)
MetricsCollector = shared_metrics.MetricsCollector
```

### 2. Initialize MetricsCollector

After loading your data and before training:

```python
# Initialize metrics collector
metrics_collector = MetricsCollector(
    model_name='YOUR_MODEL_NAME',  # e.g., 'SAGOG', 'GTWIDL', 'MPTSNet'
    dataset=dataset,                # Dataset variable
    task_type='HAR',                # or 'TSC'
    save_dir='results'
)
```

### 3. Wrap Training Loop

Wrap your training loop with the training tracker:

```python
# Start tracking training time
with metrics_collector.track_training():
    for epoch in range(num_epochs):
        # Track each epoch
        with metrics_collector.track_training_epoch():
            # Your training code here
            train_loss, train_acc = train(...)

        # Your evaluation code here
        val_loss, val_acc = validate(...)

        # Record epoch metrics
        metrics_collector.record_epoch_metrics(
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc
        )

        # Rest of your training loop (model saving, early stopping, etc.)
        # ...
```

### 4. Track Inference and Compute Final Metrics

After training completes and you load the best model:

```python
# Track inference time
with metrics_collector.track_inference():
    y_pred = infer(eval_queue, model, criterion)

# Compute throughput
metrics_collector.compute_throughput(len(test_data), phase='inference')

# Compute classification metrics
metrics_collector.compute_classification_metrics(y_true, y_pred)

# Compute model complexity
# Adjust input_shape based on your model's expected input format
input_shape = (1, input_nc, segment_size)  # Common format
metrics_collector.compute_model_complexity(model, input_shape, device=device)
```

### 5. Save Metrics

At the end of your script:

```python
# Save comprehensive metrics
print("\n" + "="*70)
print("SAVING COMPREHENSIVE METRICS")
print("="*70)
metrics_collector.save_metrics()
metrics_collector.print_summary()
```

## Model-Specific Input Shapes

Different models may have different input shape requirements for FLOPs calculation:

- **SAGoG**: `(1, input_nc, 1, segment_size)` - expects 4D input
- **MPTSNet**: `(1, segment_size, input_nc)` - expects (batch, time, features)
- **MSDL**: `(1, input_nc, segment_size)` - expects (batch, channels, time)
- **GTWIDL**: No FLOPs calculation (dictionary-based, use None for input_shape)

## Complete Example

See `codes/SAGOG/run_har_experiments.py` for a complete working example.

## Output Files

The MetricsCollector will create:
- `results/{MODEL_NAME}/{DATASET}_metrics.json` - Structured JSON with all metrics
- `results/{MODEL_NAME}/{DATASET}_summary.txt` - Human-readable summary

## Metrics Collected

### Effectiveness Metrics
- Accuracy
- Precision (macro, weighted, micro)
- Recall (macro, weighted, micro)
- F1-score (macro, weighted, micro)
- Confusion Matrix
- Per-class metrics

### Efficiency Metrics
- Training time (seconds, minutes)
- Average epoch time
- Inference time
- Throughput (samples/second)
- Peak GPU memory (GB)
- Total parameters
- Trainable parameters
- FLOPs (formatted)

### Training History
- Per-epoch training loss
- Per-epoch validation loss
- Per-epoch training accuracy
- Per-epoch validation accuracy
- Per-epoch time
