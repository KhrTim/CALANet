"""
Master script to run all TSC experiments
Runs SAGOG, GTWIDL, MPTSNet, MSDL on all TSC datasets
"""

import subprocess
import os
import sys
import argparse
from datetime import datetime

# TSC datasets
TSC_DATASETS = ["AtrialFibrillation", "MotorImagery", "Heartbeat", "PhonemeSpectra", "LSST", "PEMS-SF"]

# Models
MODELS = {
    "SAGOG": "codes/SAGOG/run_tsc_experiments.py",
    "GTWIDL": "codes/GTWIDL/run_tsc_experiments.py",
    "MPTSNet": "codes/MPTSNet/run_tsc_experiments.py",
    "MSDL": "codes/MSDL/run_tsc_experiments.py"
}

def run_experiment(model_name, dataset, gpu_id=0):
    """Run a single experiment"""
    script_path = MODELS[model_name]

    print(f"\n{'='*80}")
    print(f"Running {model_name} on {dataset} (GPU {gpu_id})")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Modify the script to use the specified dataset
    with open(script_path, 'r') as f:
        script_content = f.read()

    # Create a temporary modified script
    temp_script = script_path.replace('.py', f'_temp_{dataset}.py')

    # Update dataset selection
    lines = script_content.split('\n')
    modified_lines = []
    in_dataset_section = False

    for line in lines:
        if '# Dataset selection' in line or 'dataset =' in line:
            in_dataset_section = True

        if in_dataset_section and line.strip().startswith('#dataset ='):
            # Comment out all dataset lines
            modified_lines.append(line)
        elif in_dataset_section and line.strip().startswith('dataset ='):
            # Comment out existing uncommented dataset
            if dataset not in line:
                modified_lines.append('#' + line)
            else:
                modified_lines.append(line)
        else:
            modified_lines.append(line)
            if in_dataset_section and (line.strip() == '' or 'TSC_INFO' in line or 'input_nc' in line):
                in_dataset_section = False

    # If dataset wasn't found, add it
    if f'dataset = "{dataset}"' not in '\n'.join(modified_lines):
        for i, line in enumerate(modified_lines):
            if '# Dataset selection' in line:
                # Find the end of commented datasets
                j = i + 1
                while j < len(modified_lines) and modified_lines[j].strip().startswith('#dataset'):
                    j += 1
                # Insert the active dataset
                modified_lines.insert(j, f'dataset = "{dataset}"')
                break

    with open(temp_script, 'w') as f:
        f.write('\n'.join(modified_lines))

    # Set GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Run the experiment
    try:
        result = subprocess.run(
            [sys.executable, temp_script],
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )

        print(f"\n{'='*80}")
        print(f"Completed {model_name} on {dataset}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Exit code: {result.returncode}")
        print(f"{'='*80}\n")

        # Save output
        log_dir = f"logs_tsc/{model_name}"
        os.makedirs(log_dir, exist_ok=True)

        with open(f"{log_dir}/{dataset}_log.txt", 'w') as f:
            f.write(f"STDOUT:\n{result.stdout}\n\n")
            f.write(f"STDERR:\n{result.stderr}\n")

        # Clean up temp script
        if os.path.exists(temp_script):
            os.remove(temp_script)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"ERROR: {model_name} on {dataset} timed out after 1 hour")
        if os.path.exists(temp_script):
            os.remove(temp_script)
        return False
    except Exception as e:
        print(f"ERROR running {model_name} on {dataset}: {e}")
        if os.path.exists(temp_script):
            os.remove(temp_script)
        return False

def main():
    parser = argparse.ArgumentParser(description='Run all TSC experiments')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()) + ['all'],
                        default=['all'], help='Models to run')
    parser.add_argument('--datasets', nargs='+', choices=TSC_DATASETS + ['all'],
                        default=['all'], help='Datasets to run')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')

    args = parser.parse_args()

    # Determine which models and datasets to run
    models_to_run = list(MODELS.keys()) if 'all' in args.models else args.models
    datasets_to_run = TSC_DATASETS if 'all' in args.datasets else args.datasets

    print(f"{'='*80}")
    print(f"TSC EXPERIMENTS - Master Runner")
    print(f"{'='*80}")
    print(f"Models: {models_to_run}")
    print(f"Datasets: {datasets_to_run}")
    print(f"GPU: {args.gpu}")
    print(f"Total experiments: {len(models_to_run) * len(datasets_to_run)}")
    print(f"{'='*80}\n")

    # Track results
    results = {}
    total = len(models_to_run) * len(datasets_to_run)
    completed = 0

    # Run experiments
    for model in models_to_run:
        results[model] = {}
        for dataset in datasets_to_run:
            success = run_experiment(model, dataset, args.gpu)
            results[model][dataset] = 'SUCCESS' if success else 'FAILED'
            completed += 1
            print(f"\nProgress: {completed}/{total} experiments completed\n")

    # Print summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")

    for model in models_to_run:
        print(f"{model}:")
        for dataset in datasets_to_run:
            status = results[model][dataset]
            print(f"  {dataset:20s}: {status}")
        print()

    # Save summary
    with open('tsc_experiments_summary.txt', 'w') as f:
        f.write("TSC Experiments Summary\n")
        f.write("="*80 + "\n\n")
        for model in models_to_run:
            f.write(f"{model}:\n")
            for dataset in datasets_to_run:
                status = results[model][dataset]
                f.write(f"  {dataset:20s}: {status}\n")
            f.write("\n")

    print("Summary saved to tsc_experiments_summary.txt")

if __name__ == "__main__":
    main()
