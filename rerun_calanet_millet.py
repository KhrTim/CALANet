#!/usr/bin/env python3
"""
Re-run CALANet and millet experiments to collect training time/memory metrics
"""

import subprocess
import os
import sys
from datetime import datetime

# Experiments to re-run
EXPERIMENTS = [
    # CALANet HAR (6 datasets)
    ("CALANet", "UCI_HAR", "HAR", "codes/CALANet_local/run.py"),
    ("CALANet", "DSADS", "HAR", "codes/CALANet_local/run.py"),
    ("CALANet", "OPPORTUNITY", "HAR", "codes/CALANet_local/run.py"),
    ("CALANet", "KU-HAR", "HAR", "codes/CALANet_local/run.py"),
    ("CALANet", "PAMAP2", "HAR", "codes/CALANet_local/run.py"),
    ("CALANet", "REALDISP", "HAR", "codes/CALANet_local/run.py"),

    # CALANet TSC (6 datasets)
    ("CALANet", "AtrialFibrillation", "TSC", "codes/CALANet_local/run_TSC.py"),
    ("CALANet", "Heartbeat", "TSC", "codes/CALANet_local/run_TSC.py"),
    ("CALANet", "LSST", "TSC", "codes/CALANet_local/run_TSC.py"),
    ("CALANet", "MotorImagery", "TSC", "codes/CALANet_local/run_TSC.py"),
    ("CALANet", "PEMS-SF", "TSC", "codes/CALANet_local/run_TSC.py"),
    ("CALANet", "PhonemeSpectra", "TSC", "codes/CALANet_local/run_TSC.py"),

    # millet HAR (5 datasets - excluding REALDISP which already failed)
    ("millet", "UCI_HAR", "HAR", "codes/millet/run.py"),
    ("millet", "DSADS", "HAR", "codes/millet/run.py"),
    ("millet", "OPPORTUNITY", "HAR", "codes/millet/run.py"),
    ("millet", "KU-HAR", "HAR", "codes/millet/run.py"),
    ("millet", "PAMAP2", "HAR", "codes/millet/run.py"),

    # millet TSC (6 datasets)
    ("millet", "AtrialFibrillation", "TSC", "codes/millet/run_TSC.py"),
    ("millet", "Heartbeat", "TSC", "codes/millet/run_TSC.py"),
    ("millet", "LSST", "TSC", "codes/millet/run_TSC.py"),
    ("millet", "MotorImagery", "TSC", "codes/millet/run_TSC.py"),
    ("millet", "PEMS-SF", "TSC", "codes/millet/run_TSC.py"),
    ("millet", "PhonemeSpectra", "TSC", "codes/millet/run_TSC.py"),
]

def run_experiment(model, dataset, task_type, script_path):
    """Run a single experiment"""
    print(f"\n{'='*80}")
    print(f"Running {model} on {dataset} ({task_type})")
    print(f"{'='*80}")

    # Create temp script with dataset set
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)
    temp_script = os.path.join(script_dir, f'temp_{dataset}_rerun_{script_name}')

    try:
        # Read original script
        with open(script_path, 'r') as f:
            content = f.read()

        # Modify dataset line
        lines = content.split('\n')
        modified_lines = []
        for line in lines:
            if line.strip().startswith('dataset = '):
                # Comment out existing dataset lines
                if not line.strip().startswith('#'):
                    modified_lines.append('#' + line)
                else:
                    modified_lines.append(line)
            else:
                modified_lines.append(line)

        # Add dataset specification after imports
        insert_pos = 0
        for i, line in enumerate(modified_lines):
            if 'MetricsCollector = shared_metrics.MetricsCollector' in line:
                insert_pos = i + 1
                break

        modified_lines.insert(insert_pos, f'\n# Dataset specified by rerun script\ndataset = "{dataset}"\n')

        # Write temp script
        with open(temp_script, 'w') as f:
            f.write('\n'.join(modified_lines))

        # Use the rthar conda environment Python (has PyTorch installed)
        python_executable = '/userHome/userhome1/timur/miniconda3/envs/rthar/bin/python'
        if not os.path.exists(python_executable):
            # Fallback to sys.executable if rthar env doesn't exist
            python_executable = sys.executable

        # Run experiment
        start_time = datetime.now()
        result = subprocess.run(
            [python_executable, temp_script],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per experiment
        )
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        # Save log
        log_dir = f"logs_{task_type.lower()}/{model}"
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/{dataset}_log.txt", 'w') as f:
            f.write(f"Command: {python_executable} {temp_script}\n")
            f.write(f"Start: {start_time}\n")
            f.write(f"End: {end_time}\n")
            f.write(f"Duration: {elapsed:.1f}s\n")
            f.write(f"Exit Code: {result.returncode}\n\n")
            f.write("="*80 + "\n")
            f.write("STDOUT:\n")
            f.write("="*80 + "\n")
            f.write(result.stdout)
            f.write("\n\n")
            f.write("="*80 + "\n")
            f.write("STDERR:\n")
            f.write("="*80 + "\n")
            f.write(result.stderr)

        # Check if successful
        success = result.returncode == 0
        metrics_file = f"results/{model}/{dataset}_metrics.json"
        has_metrics = os.path.exists(metrics_file)

        if success and has_metrics:
            print(f"✅ SUCCESS: {model} on {dataset} ({elapsed:.1f}s)")
            return True
        elif success and not has_metrics:
            print(f"⚠️  WARNING: {model} on {dataset} completed but no metrics file ({elapsed:.1f}s)")
            return False
        else:
            print(f"❌ FAILED: {model} on {dataset} ({elapsed:.1f}s)")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT: {model} on {dataset}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {model} on {dataset}: {e}")
        return False
    finally:
        # Clean up temp script
        if os.path.exists(temp_script):
            os.remove(temp_script)

def main():
    print("="*80)
    print("RE-RUNNING CALANet AND millet EXPERIMENTS")
    print("="*80)
    print(f"Total experiments to run: {len(EXPERIMENTS)}")
    print()

    results = []
    for i, (model, dataset, task_type, script_path) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] Starting {model} on {dataset} ({task_type})...")
        success = run_experiment(model, dataset, task_type, script_path)
        results.append((model, dataset, task_type, success))

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    successful = sum(1 for _, _, _, s in results if s)
    failed = len(results) - successful

    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    if failed > 0:
        print("Failed experiments:")
        for model, dataset, task_type, success in results:
            if not success:
                print(f"  - {model} on {dataset} ({task_type})")

    print("="*80)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
