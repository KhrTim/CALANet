"""
Batch script to add metrics collection to all experiment scripts.

This script systematically updates all run_har_experiments.py and run_tsc_experiments.py
files to integrate the comprehensive metrics collection framework.
"""

import os
import re
from pathlib import Path


def find_experiment_scripts():
    """Find all HAR and TSC experiment scripts."""
    scripts = []
    codes_dir = Path('codes')

    for model_dir in codes_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # Skip virtual environments and cache
        if model_dir.name.startswith('.') or model_dir.name == '__pycache__':
            continue

        # Look for experiment scripts
        har_script = model_dir / 'run_har_experiments.py'
        tsc_script = model_dir / 'run_tsc_experiments.py'

        if har_script.exists():
            scripts.append((str(har_script), model_dir.name, 'HAR'))
        if tsc_script.exists():
            scripts.append((str(tsc_script), model_dir.name, 'TSC'))

    return scripts


def has_metrics_import(content):
    """Check if script already imports MetricsCollector."""
    return 'MetricsCollector' in content or 'shared_metrics' in content


def add_metrics_import(content, model_name):
    """Add MetricsCollector import to the script."""

    # Find the import section (after existing util imports)
    # Look for patterns like "from utils import" or "import importlib"
    import_pattern = r'(import importlib\.util.*?(?:EarlyStopping|spec\.loader\.exec_module)\([^)]+\))'

    metrics_import = f"""
# Import shared metrics collector
spec = importlib.util.spec_from_file_location("shared_metrics", os.path.join(codes_dir, 'shared_metrics.py'))
shared_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_metrics)
MetricsCollector = shared_metrics.MetricsCollector
"""

    # Try to find the last importlib.util section
    matches = list(re.finditer(import_pattern, content, re.DOTALL))
    if matches:
        last_match = matches[-1]
        insert_pos = last_match.end()
        content = content[:insert_pos] + metrics_import + content[insert_pos:]
    else:
        # Fallback: add after imports section
        # Find line after last import statement
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                insert_idx = i + 1

        lines.insert(insert_idx, metrics_import)
        content = '\n'.join(lines)

    return content


def add_metrics_initialization(content, model_name, task_type):
    """Add metrics collector initialization."""

    # Find the best place to add initialization
    # Look for patterns like "# Early stopping", "early_stopping = ", "# Training loop"

    init_code = f"""
# Initialize metrics collector
metrics_collector = MetricsCollector(
    model_name='{model_name}',
    dataset=dataset,
    task_type='{task_type}',
    save_dir='results'
)
"""

    # Try to find early_stopping or training loop start
    patterns = [
        r'(# Early stopping\s+early_stopping = [^\n]+)',
        r'(early_stopping = EarlyStopping[^\n]+)',
        r'(# Training loop)'
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + '\n' + init_code + content[insert_pos:]
            return content

    # Fallback: add before first "for epoch" loop
    match = re.search(r'(\nfor epoch in range)', content)
    if match:
        insert_pos = match.start()
        content = content[:insert_pos] + '\n' + init_code + content[insert_pos:]

    return content


def add_metrics_to_training_loop(content):
    """Wrap training loop with metrics tracking."""

    # This is complex and model-specific, so we'll just add comments/reminders
    # Manual adjustment will be needed for proper integration

    # Add a reminder comment before the training loop
    reminder = """
# TODO: Wrap training loop with metrics_collector.track_training()
# TODO: Wrap each epoch with metrics_collector.track_training_epoch()
# TODO: Call metrics_collector.record_epoch_metrics() after each epoch
"""

    match = re.search(r'(\n# Training loop|\nfor epoch in range)', content)
    if match:
        insert_pos = match.start()
        content = content[:insert_pos] + '\n' + reminder + content[insert_pos:]

    return content


def add_final_metrics_collection(content):
    """Add final metrics collection at the end."""

    final_code = """
# Compute comprehensive metrics
print("\\n" + "="*70)
print("COMPUTING COMPREHENSIVE METRICS")
print("="*70)

# TODO: Wrap inference with metrics_collector.track_inference()
# TODO: Compute throughput: metrics_collector.compute_throughput(len(test_data), phase='inference')
# TODO: Compute classification metrics: metrics_collector.compute_classification_metrics(y_true, y_pred)
# TODO: Compute model complexity: metrics_collector.compute_model_complexity(model, input_shape, device=device)

# Save comprehensive metrics
metrics_collector.save_metrics()
metrics_collector.print_summary()
"""

    # Add at the very end of the file, or before the last print statement
    # Look for patterns like "Results saved to" or end of file
    match = re.search(r'(print\(f".*Results saved to)', content)
    if match:
        insert_pos = match.end()
        # Find the end of that line
        newline_pos = content.find('\n', insert_pos)
        if newline_pos != -1:
            content = content[:newline_pos+1] + final_code + content[newline_pos+1:]
    else:
        # Add at end
        content = content.rstrip() + '\n\n' + final_code

    return content


def update_script(filepath, model_name, task_type, dry_run=True):
    """Update a single experiment script."""

    print(f"\n{'='*70}")
    print(f"Processing: {filepath}")
    print(f"Model: {model_name}, Task: {task_type}")
    print(f"{'='*70}")

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        original_content = content

        # Check if already has metrics
        if has_metrics_import(content):
            print("⚠️  Script already has MetricsCollector import, skipping...")
            return False

        # Apply modifications
        print("✓ Adding MetricsCollector import...")
        content = add_metrics_import(content, model_name)

        print("✓ Adding metrics initialization...")
        content = add_metrics_initialization(content, model_name, task_type)

        print("✓ Adding training loop reminders...")
        content = add_metrics_to_training_loop(content)

        print("✓ Adding final metrics collection...")
        content = add_final_metrics_collection(content)

        if dry_run:
            print("\n📝 DRY RUN - Changes not saved")
            print(f"Would modify {len(content) - len(original_content)} characters")

            # Save to a temp file for review
            temp_file = filepath.replace('.py', '_PREVIEW.py')
            with open(temp_file, 'w') as f:
                f.write(content)
            print(f"Preview saved to: {temp_file}")
        else:
            # Create backup
            backup_file = filepath + '.backup'
            with open(backup_file, 'w') as f:
                f.write(original_content)
            print(f"Backup saved to: {backup_file}")

            # Write updated content
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Updated: {filepath}")

        return True

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch add metrics to experiment scripts')
    parser.add_argument('--dry-run', action='store_true', default=True,
                      help='Preview changes without modifying files (default: True)')
    parser.add_argument('--apply', action='store_true',
                      help='Actually apply changes (overrides --dry-run)')
    parser.add_argument('--model', type=str,
                      help='Only process specific model')
    parser.add_argument('--task', choices=['HAR', 'TSC'],
                      help='Only process HAR or TSC scripts')

    args = parser.parse_args()

    dry_run = not args.apply

    print("="*70)
    print("BATCH METRICS INTEGRATION")
    print("="*70)
    print(f"Mode: {'DRY RUN (preview only)' if dry_run else 'APPLY CHANGES'}")
    print("="*70)

    # Find all scripts
    scripts = find_experiment_scripts()

    # Filter by model and task if specified
    if args.model:
        scripts = [(s, m, t) for s, m, t in scripts if m == args.model]
    if args.task:
        scripts = [(s, m, t) for s, m, t in scripts if t == args.task]

    print(f"\nFound {len(scripts)} experiment scripts to process:")
    for filepath, model, task in scripts:
        print(f"  - {model}/{task}: {filepath}")

    if not scripts:
        print("No scripts found to process.")
        return

    # Process each script
    print("\n" + "="*70)
    print("PROCESSING SCRIPTS")
    print("="*70)

    updated_count = 0
    for filepath, model, task in scripts:
        if update_script(filepath, model, task, dry_run=dry_run):
            updated_count += 1

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total scripts: {len(scripts)}")
    print(f"Updated: {updated_count}")
    print(f"Skipped: {len(scripts) - updated_count}")

    if dry_run:
        print("\n⚠️  This was a DRY RUN - no files were modified")
        print("Review the *_PREVIEW.py files and run with --apply to apply changes")
    else:
        print("\n✅ Files updated successfully")
        print("Backup files created with .backup extension")
        print("\n⚠️  IMPORTANT: Review the TODO comments in each file")
        print("Some manual adjustments may be needed for proper integration")


if __name__ == '__main__':
    main()
