"""
Multi-GPU Parallel Experiment Runner
Distributes experiments across multiple GPUs for faster execution
"""

import subprocess
import os
import sys
import argparse
from datetime import datetime
import time
from multiprocessing import Process, Queue

# All datasets
HAR_DATASETS = ["UCI_HAR", "DSADS", "OPPORTUNITY", "KU-HAR", "PAMAP2", "REALDISP"]
TSC_DATASETS = ["AtrialFibrillation", "MotorImagery", "Heartbeat", "PhonemeSpectra", "LSST", "PEMS-SF"]

# Models
MODELS = ["SAGOG", "GTWIDL", "MPTSNet", "MSDL"]

def worker(task_queue, result_queue, gpu_id):
    """Worker process that runs experiments on a specific GPU"""
    while True:
        task = task_queue.get()

        if task is None:  # Poison pill to stop worker
            break

        model, dataset, dataset_type = task

        print(f"\n[GPU {gpu_id}] Starting {model} on {dataset} ({dataset_type})")
        start_time = time.time()

        # Determine script to run
        if dataset_type == 'HAR':
            master_script = 'run_all_har_experiments.py'
        else:
            master_script = 'run_all_tsc_experiments.py'

        # Run experiment
        try:
            result = subprocess.run(
                [sys.executable, master_script,
                 '--models', model,
                 '--datasets', dataset,
                 '--gpu', str(gpu_id)],
                capture_output=True,
                text=True,
                timeout=3600
            )

            elapsed = time.time() - start_time
            success = result.returncode == 0

            print(f"[GPU {gpu_id}] Completed {model} on {dataset} - {'SUCCESS' if success else 'FAILED'} ({elapsed:.1f}s)")

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

def main():
    parser = argparse.ArgumentParser(description='Run experiments in parallel across multiple GPUs')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='GPU IDs to use (e.g., --gpus 0 1 2 3)')
    parser.add_argument('--split', choices=['model', 'dataset'], default='dataset',
                        help='How to split work: by model or by dataset')
    parser.add_argument('--har', action='store_true', help='Run HAR experiments')
    parser.add_argument('--tsc', action='store_true', help='Run TSC experiments')

    args = parser.parse_args()

    # Default to running both if neither specified
    if not args.har and not args.tsc:
        args.har = True
        args.tsc = True

    num_gpus = len(args.gpus)

    print(f"{'='*80}")
    print(f"PARALLEL EXPERIMENT RUNNER")
    print(f"{'='*80}")
    print(f"GPUs: {args.gpus} ({num_gpus} GPUs)")
    print(f"Split strategy: {args.split}")
    print(f"HAR experiments: {args.har}")
    print(f"TSC experiments: {args.tsc}")
    print(f"{'='*80}\n")

    # Create task list
    tasks = []

    if args.har:
        for model in MODELS:
            for dataset in HAR_DATASETS:
                tasks.append((model, dataset, 'HAR'))

    if args.tsc:
        for model in MODELS:
            for dataset in TSC_DATASETS:
                tasks.append((model, dataset, 'TSC'))

    total_tasks = len(tasks)
    print(f"Total experiments to run: {total_tasks}")
    print(f"Estimated time with {num_gpus} GPUs: ~{(total_tasks / num_gpus) * 20} minutes")
    print(f"(assuming ~20 min per experiment)\n")

    # Optionally reorder tasks based on split strategy
    if args.split == 'model':
        # Group by model to run all datasets for one model on same GPU
        tasks.sort(key=lambda x: (x[0], x[2], x[1]))
    else:
        # Group by dataset to run all models for one dataset on same GPU
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
        p = Process(target=worker, args=(task_queue, result_queue, gpu_id))
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

    # Group by model and dataset
    by_model = {}
    by_dataset = {}

    for r in results:
        model = r['model']
        dataset = r['dataset']
        status = 'SUCCESS' if r['success'] else 'FAILED'

        if model not in by_model:
            by_model[model] = {}
        by_model[model][dataset] = status

        if dataset not in by_dataset:
            by_dataset[dataset] = {}
        by_dataset[dataset][model] = status

    # Print by model
    print("By Model:")
    for model in MODELS:
        if model in by_model:
            successes = sum(1 for v in by_model[model].values() if v == 'SUCCESS')
            total = len(by_model[model])
            print(f"  {model}: {successes}/{total} successful")

    print("\nBy Dataset:")
    all_datasets = (HAR_DATASETS if args.har else []) + (TSC_DATASETS if args.tsc else [])
    for dataset in all_datasets:
        if dataset in by_dataset:
            successes = sum(1 for v in by_dataset[dataset].values() if v == 'SUCCESS')
            total = len(by_dataset[dataset])
            print(f"  {dataset:20s}: {successes}/{total} successful")

    # Save detailed summary
    with open('parallel_experiments_summary.txt', 'w') as f:
        f.write("Parallel Experiments Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total: {total_tasks}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Failed: {failed_count}\n\n")

        f.write("="*80 + "\n")
        f.write("By Model:\n")
        f.write("="*80 + "\n")
        for model in MODELS:
            if model in by_model:
                f.write(f"\n{model}:\n")
                for dataset, status in sorted(by_model[model].items()):
                    f.write(f"  {dataset:20s}: {status}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("By Dataset:\n")
        f.write("="*80 + "\n")
        for dataset in all_datasets:
            if dataset in by_dataset:
                f.write(f"\n{dataset}:\n")
                for model, status in sorted(by_dataset[dataset].items()):
                    f.write(f"  {model:10s}: {status}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Failed Experiments:\n")
        f.write("="*80 + "\n")
        for r in results:
            if not r['success']:
                error = r.get('error', 'Unknown')
                f.write(f"  {r['model']:10s} on {r['dataset']:20s}: {error}\n")

    print("\nDetailed summary saved to parallel_experiments_summary.txt")

if __name__ == "__main__":
    main()
