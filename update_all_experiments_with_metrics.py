"""
Automated script to add comprehensive metrics collection to all experiment scripts.

This script systematically updates all run*.py files to integrate the MetricsCollector.
"""

import os
import re
from pathlib import Path


# Model configurations with their specific details
MODEL_CONFIGS = {
    # HAR models
    'RepHAR': {'task': 'HAR', 'script': 'run.py', 'input_format': '3d'},
    'DeepConvLSTM': {'task': 'HAR', 'script': 'run.py', 'input_format': '3d'},
    'Bi-GRU-I': {'task': 'HAR', 'script': 'run.py', 'input_format': '3d'},
    'RevTransformerAttentionHAR': {'task': 'HAR', 'script': 'run.py', 'input_format': '3d'},
    'IF-ConvTransformer2': {'task': 'HAR', 'script': 'run.py', 'input_format': '3d'},
    'millet': {'task': 'BOTH', 'script_har': 'run.py', 'script_tsc': 'run_TSC.py', 'input_format': '3d'},
    'DSN-master': {'task': 'BOTH', 'script_har': 'run.py', 'script_tsc': 'run_TSC.py', 'input_format': '3d'},
    'SAGOG': {'task': 'BOTH', 'script_har': 'run_har_experiments.py', 'script_tsc': 'run_tsc_experiments.py', 'input_format': '4d'},
    'MPTSNet': {'task': 'BOTH', 'script_har': 'run_har_experiments.py', 'script_tsc': 'run_tsc_experiments.py', 'input_format': 'time_first'},
    'MSDL': {'task': 'BOTH', 'script_har': 'run_har_experiments.py', 'script_tsc': 'run_tsc_experiments.py', 'input_format': '3d'},
    'GTWIDL': {'task': 'BOTH', 'script_har': 'run_har_experiments.py', 'script_tsc': 'run_tsc_experiments.py', 'input_format': 'dict'},

    # TSC models
    'resnet': {'task': 'TSC', 'script': 'run_TSC.py', 'input_format': '3d'},
    'FCN_TSC': {'task': 'TSC', 'script': 'run_TSC.py', 'input_format': '3d'},
    'InceptionTime': {'task': 'TSC', 'script': 'run_TSC.py', 'input_format': '3d'},
    'TapNet': {'task': 'TSC', 'script': 'run_TSC.py', 'input_format': '3d'},
}


def get_metrics_import_code():
    """Get the code to import MetricsCollector."""
    return """
# Import shared metrics collector
import importlib.util
codes_dir_for_metrics = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("shared_metrics",
                                              os.path.join(codes_dir_for_metrics, 'shared_metrics.py'))
shared_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_metrics)
MetricsCollector = shared_metrics.MetricsCollector
"""


def get_metrics_init_code(model_name, task_type):
    """Get the code to initialize MetricsCollector."""
    return f"""
# Initialize metrics collector
metrics_collector = MetricsCollector(
    model_name='{model_name}',
    dataset=dataset,
    task_type='{task_type}',
    save_dir='results'
)
"""


def get_input_shape_code(model_name, input_format):
    """Get input shape based on model format."""
    if input_format == '4d':
        return "(1, input_nc, 1, segment_size)"
    elif input_format == 'time_first':
        return "(1, segment_size, input_nc)"
    elif input_format == '3d':
        return "(1, input_nc, segment_size)"
    elif input_format == 'dict':
        return "None  # Dictionary-based model"
    else:
        return "(1, input_nc, segment_size)"


def update_script(filepath, model_name, task_type):
    """Update a single experiment script with metrics collection."""

    print(f"\nUpdating: {filepath}")

    if not os.path.exists(filepath):
        print(f"  ⚠️  File not found, skipping")
        return False

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Check if already updated
        if 'MetricsCollector' in content:
            print(f"  ✓ Already has MetricsCollector, skipping")
            return False

        original_content = content

        # Step 1: Add import after existing imports
        # Find the last import statement
        import_lines = []
        lines = content.split('\n')
        last_import_idx = 0

        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                last_import_idx = i

        # Insert metrics import after last import
        lines.insert(last_import_idx + 1, get_metrics_import_code())
        content = '\n'.join(lines)

        # Step 2: Add metrics initialization before training loop
        # Look for common patterns: "for epoch in range", "model.train()", training loop markers

        # Try to find where the model is created and training starts
        # Common patterns: "model = ", "criterion = ", "optimizer = "

        init_code = get_metrics_init_code(model_name, task_type)

        # Insert after optimizer/criterion setup but before training loop
        patterns = [
            (r'(optimizer = .*\n)', lambda m: m.group(1) + init_code),
            (r'(criterion = .*\.cuda\(\)\n)', lambda m: m.group(1) + init_code),
            (r'(# Set random seeds.*?\n.*?device = .*\n)', lambda m: m.group(1) + init_code),
        ]

        inserted = False
        for pattern, replacement in patterns:
            if re.search(pattern, content, re.DOTALL):
                content = re.sub(pattern, replacement, content, count=1)
                inserted = True
                break

        if not inserted:
            # Fallback: insert before first "for epoch" or "model.train()"
            match = re.search(r'(\nfor epoch in range)', content)
            if match:
                pos = match.start()
                content = content[:pos] + '\n' + init_code + content[pos:]
                inserted = True

        if not inserted:
            print(f"  ⚠️  Could not find insertion point for metrics init")
            # Still continue to add other parts

        # Step 3: Wrap training loop
        # Find "for epoch in range" and wrap with track_training()

        # This is complex - we need to identify the training loop scope
        # For now, add commented instructions
        training_wrapper_comment = """
# TODO: Wrap training loop with metrics_collector.track_training()
# Example:
# with metrics_collector.track_training():
#     for epoch in range(epoches):
#         with metrics_collector.track_training_epoch():
#             train_loss, train_acc = train(...)
#         metrics_collector.record_epoch_metrics(train_loss=train_loss, val_loss=val_loss,
#                                               train_acc=train_acc, val_acc=val_acc)
"""

        # Add comment before training loop
        content = re.sub(r'(\nfor epoch in range)',
                        '\n' + training_wrapper_comment + r'\1',
                        content, count=1)

        # Step 4: Add final metrics collection at the end
        input_format = MODEL_CONFIGS.get(model_name, {}).get('input_format', '3d')
        input_shape = get_input_shape_code(model_name, input_format)

        final_metrics_code = f"""

# ============================================================================
# COMPREHENSIVE METRICS COLLECTION
# ============================================================================
print("\\n" + "="*70)
print("COLLECTING COMPREHENSIVE METRICS")
print("="*70)

# TODO: Wrap inference with metrics_collector.track_inference()
# Example:
# with metrics_collector.track_inference():
#     eval_loss, y_pred = infer(eval_queue, model, criterion)

# TODO: Add these lines after getting predictions:
# y_pred_labels = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else y_pred
# metrics_collector.compute_throughput(len(y_test_unary), phase='inference')
# metrics_collector.compute_classification_metrics(y_test_unary, y_pred_labels)
#
# # Compute model complexity
# input_shape = {input_shape}
# if input_shape is not None:
#     metrics_collector.compute_model_complexity(model, input_shape, device='cuda')
#
# # Save comprehensive metrics
# metrics_collector.save_metrics()
# metrics_collector.print_summary()
"""

        # Add at the end of file
        content = content.rstrip() + '\n' + final_metrics_code + '\n'

        # Create backup
        backup_file = filepath + '.backup'
        with open(backup_file, 'w') as f:
            f.write(original_content)

        # Write updated content
        with open(filepath, 'w') as f:
            f.write(content)

        print(f"  ✅ Updated successfully")
        print(f"  📁 Backup saved to: {backup_file}")
        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    print("="*70)
    print("AUTOMATIC METRICS INTEGRATION FOR ALL MODELS")
    print("="*70)

    codes_dir = Path('codes')
    updated_count = 0
    skipped_count = 0
    failed_count = 0

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Processing: {model_name}")
        print(f"{'='*70}")

        task = config['task']

        if task == 'BOTH':
            # Update both HAR and TSC scripts
            har_script = codes_dir / model_name / config['script_har']
            tsc_script = codes_dir / model_name / config['script_tsc']

            if har_script.exists():
                if update_script(str(har_script), model_name, 'HAR'):
                    updated_count += 1
                else:
                    skipped_count += 1
            else:
                print(f"  ⚠️  HAR script not found: {har_script}")
                failed_count += 1

            if tsc_script.exists():
                if update_script(str(tsc_script), model_name, 'TSC'):
                    updated_count += 1
                else:
                    skipped_count += 1
            else:
                print(f"  ⚠️  TSC script not found: {tsc_script}")
                failed_count += 1

        else:
            # Single task model
            script_name = config.get('script')
            if script_name:
                script_path = codes_dir / model_name / script_name

                if script_path.exists():
                    if update_script(str(script_path), model_name, task):
                        updated_count += 1
                    else:
                        skipped_count += 1
                else:
                    print(f"  ⚠️  Script not found: {script_path}")
                    failed_count += 1

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Updated: {updated_count}")
    print(f"Skipped (already updated): {skipped_count}")
    print(f"Failed: {failed_count}")
    print()
    print("⚠️  IMPORTANT: Review the TODO comments in each file!")
    print("Some manual adjustments may be needed for proper integration.")
    print()
    print("Next steps:")
    print("1. Review the updated files and complete the TODO sections")
    print("2. Test on a single model to verify it works")
    print("3. Run full experiments with new metrics collection")


if __name__ == '__main__':
    main()
