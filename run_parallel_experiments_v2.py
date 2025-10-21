"""
Multi-GPU Parallel Experiment Runner (Fixed Version)
Directly runs individual experiment scripts in parallel across GPUs
"""

import subprocess
import os
import sys
import argparse
from datetime import datetime
import time
from multiprocessing import Process, Queue
import re

# All datasets
HAR_DATASETS = ["UCI_HAR", "DSADS", "OPPORTUNITY", "KU-HAR", "PAMAP2", "REALDISP"]
TSC_DATASETS = ["AtrialFibrillation", "MotorImagery", "Heartbeat", "PhonemeSpectra", "LSST", "PEMS-SF"]

# Model scripts
MODEL_SCRIPTS = {
    'HAR': {
        'SAGOG': 'codes/SAGOG/run_har_experiments.py',
        'GTWIDL': 'codes/GTWIDL/run_har_experiments.py',
        'MPTSNet': 'codes/MPTSNet/run_har_experiments.py',
        'MSDL': 'codes/MSDL/run_har_experiments.py'
    },
    'TSC': {
        'SAGOG': 'codes/SAGOG/run_tsc_experiments.py',
        'GTWIDL': 'codes/GTWIDL/run_tsc_experiments.py',
        'MPTSNet': 'codes/MPTSNet/run_tsc_experiments.py',
        'MSDL': 'codes/MSDL/run_tsc_experiments.py'
    }
}

def modify_script_dataset(script_path, target_dataset):
    """Modify script to use target dataset without creating temp file"""
    with open(script_path, 'r') as f:
        content = f.read()

    # Replace dataset assignment
    lines = content.split('\n')
    modified = []

    for line in lines:
        # Comment out all dataset assignments
        if line.strip().startswith('dataset =') and '#' not in line.split('dataset')[0]:
            # Check if this is the target dataset
            if f'"{target_dataset}"' in line or f"'{target_dataset}'" in line:
                modified.append(line)  # Keep this one uncommented
            else:
                modified.append('#' + line)  # Comment out others
        # Uncomment the target dataset if it's commented
        elif line.strip().startswith('#dataset =') and target_dataset in line:
            modified.append(line.lstrip('#'))
        else:
            modified.append(line)

    return '\n'.join(modified)

def worker(task_queue, result_queue, gpu_id, project_root):
    """Worker process that runs experiments on a specific GPU"""
    while True:
        task = task_queue.get()

        if task is None:  # Poison pill to stop worker
            break

        model, dataset, dataset_type = task

        print(f"\n[GPU {gpu_id}] Starting {model} on {dataset} ({dataset_type})")
        start_time = time.time()

        # Get script path
        script_path = os.path.join(project_root, MODEL_SCRIPTS[dataset_type][model])

        # Create modified script content
        modified_content = modify_script_dataset(script_path, dataset)

        # Create temp script in project root
        temp_script = os.path.join(project_root, f'temp_{model}_{dataset}_{gpu_id}.py')

        try:
            with open(temp_script, 'w') as f:
                f.write(modified_content)

            # Set environment
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            # Run experiment
            result = subprocess.run(
                [sys.executable, temp_script],
                env=env,
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=3600
            )

            elapsed = time.time() - start_time
            success = result.returncode == 0

            print(f"[GPU {gpu_id}] Completed {model} on {dataset} - {'SUCCESS' if success else 'FAILED'} ({elapsed:.1f}s)")

            # Save detailed log
            log_dir = os.path.join(project_root, f'logs_{dataset_type.lower()}', model)
            os.makedirs(log_dir, exist_ok=True)

            with open(os.path.join(log_dir, f'{dataset}_log.txt'), 'w') as f:
                f.write(f"Command: python {temp_script}\n")
                f.write(f"GPU: {gpu_id}\n")
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

            result_queue.put({
                'model': model,
                'dataset': dataset,
                'dataset_type': dataset_type,
                'gpu': gpu_id,
                'success': success,
                'time': elapsed
            })

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"[GPU {gpu_id}] TIMEOUT {model} on {dataset} ({elapsed:.1f}s)")

            result_queue.put({
                'model': model,
                'dataset': dataset,
                'dataset_type': dataset_type,
                'gpu': gpu_id,
                'success': False,
                'time': elapsed,
                'error': 'TIMEOUT'
            })

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[GPU {gpu_id}] ERROR {model} on {dataset}: {e}")

            result_queue.put({
                'model': model,
                'dataset': dataset,
                'dataset_type': dataset_type,
                'gpu': gpu_id,
                'success': False,
                'time': elapsed,
                'error': str(e)
            })

        finally:
            # Clean up temp script
            if os.path.exists(temp_script):
                try:
                    os.remove(temp_script)
                except:
                    pass

def main():
    parser = argparse.ArgumentParser(description='Run experiments in parallel across multiple GPUs')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='GPU IDs to use (e.g., --gpus 0 1 2 3)')
    parser.add_argument('--split', choices=['model', 'dataset'], default='dataset',
                        help='How to split work: by model or by dataset')
    parser.add_argument('--har', action='store_true', help='Run HAR experiments')
    parser.add_argument('--tsc', action='store_true', help='Run TSC experiments')
    parser.add_argument('--models', nargs='+', choices=['SAGOG', 'GTWIDL', 'MPTSNet', 'MSDL', 'all'],
                        default=['all'], help='Models to run')
    parser.add_argument('--datasets-har', nargs='+', choices=HAR_DATASETS + ['all'],
                        default=['all'], help='HAR datasets to run')
    parser.add_argument('--datasets-tsc', nargs='+', choices=TSC_DATASETS + ['all'],
                        default=['all'], help='TSC datasets to run')

    args = parser.parse_args()

    # Default to running both if neither specified
    if not args.har and not args.tsc:
        args.har = True
        args.tsc = True

    # Determine models
    if 'all' in args.models:
        models_to_run = ['SAGOG', 'GTWIDL', 'MPTSNet', 'MSDL']
    else:
        models_to_run = args.models

    # Determine datasets
    har_datasets = HAR_DATASETS if 'all' in args.datasets_har else args.datasets_har
    tsc_datasets = TSC_DATASETS if 'all' in args.datasets_tsc else args.datasets_tsc

    num_gpus = len(args.gpus)
    project_root = os.path.dirname(os.path.abspath(__file__))

    print(f"{'='*80}")
    print(f"PARALLEL EXPERIMENT RUNNER V2")
    print(f"{'='*80}")
    print(f"Project root: {project_root}")
    print(f"GPUs: {args.gpus} ({num_gpus} GPUs)")
    print(f"Models: {models_to_run}")
    print(f"Split strategy: {args.split}")
    print(f"HAR experiments: {args.har}")
    print(f"TSC experiments: {args.tsc}")
    if args.har:
        print(f"HAR datasets: {har_datasets}")
    if args.tsc:
        print(f"TSC datasets: {tsc_datasets}")
    print(f"{'='*80}\n")

    # Create task list
    tasks = []

    if args.har:
        for model in models_to_run:
            for dataset in har_datasets:
                tasks.append((model, dataset, 'HAR'))

    if args.tsc:
        for model in models_to_run:
            for dataset in tsc_datasets:
                tasks.append((model, dataset, 'TSC'))

    total_tasks = len(tasks)
    print(f"Total experiments to run: {total_tasks}")
    print(f"Estimated time with {num_gpus} GPUs: ~{(total_tasks / num_gpus) * 20} minutes")
    print(f"(assuming ~20 min per experiment)\n")

    # Optionally reorder tasks based on split strategy
    if args.split == 'model':
        tasks.sort(key=lambda x: (x[0], x[2], x[1]))
    else:
        tasks.sort(key=lambda x: (x[2], x[1], x[0]))

    # Create queues
    task_queue = Queue()
    result_queue = Queue()

    # Add tasks to queue
    for task in tasks:
        task_queue.put(task)

    # Add poison pills to stop workers
    for _ in range(num_gpus):
        task_queue.put(None)

    # Start workers
    print(f"Starting {num_gpus} worker processes...\n")
    workers = []
    for i, gpu_id in enumerate(args.gpus):
        p = Process(target=worker, args=(task_queue, result_queue, gpu_id, project_root))
        p.start()
        workers.append(p)

    # Collect results
    results = []
    for _ in range(total_tasks):
        result = result_queue.get()
        results.append(result)

    # Wait for all workers to finish
    for p in workers:
        p.join()

    # Print summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")

    success_count = sum(1 for r in results if r['success'])
    failed_count = total_tasks - success_count

    print(f"Total: {total_tasks}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print()

    # Group by model
    by_model = {}
    for r in results:
        model = r['model']
        if model not in by_model:
            by_model[model] = {'success': 0, 'total': 0}
        by_model[model]['total'] += 1
        if r['success']:
            by_model[model]['success'] += 1

    print("By Model:")
    for model in models_to_run:
        if model in by_model:
            s = by_model[model]['success']
            t = by_model[model]['total']
            print(f"  {model:10s}: {s}/{t} successful")

    # Save detailed summary
    with open(os.path.join(project_root, 'parallel_experiments_summary.txt'), 'w') as f:
        f.write("Parallel Experiments Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total: {total_tasks}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Failed: {failed_count}\n\n")

        f.write("="*80 + "\n")
        f.write("Detailed Results:\n")
        f.write("="*80 + "\n")
        for r in sorted(results, key=lambda x: (x['model'], x['dataset'])):
            status = 'SUCCESS' if r['success'] else f"FAILED ({r.get('error', 'Unknown')})"
            f.write(f"{r['model']:10s} | {r['dataset']:20s} | {r['dataset_type']:3s} | GPU{r['gpu']} | {r['time']:6.1f}s | {status}\n")

    print(f"\nDetailed summary saved to parallel_experiments_summary.txt")
    print(f"Logs saved to logs_har/ and logs_tsc/")

if __name__ == "__main__":
    main()
