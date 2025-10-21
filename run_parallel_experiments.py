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

def is_experiment_successful(model, dataset, dataset_type):
    """Check if an experiment has already completed successfully"""
    log_dir = f"logs_{dataset_type.lower()}/{model}"
    log_file = f"{log_dir}/{dataset}_log.txt"

    if not os.path.exists(log_file):
        return False

    try:
        with open(log_file, 'r') as f:
            content = f.read()
            return 'Exit Code: 0' in content
    except:
        return False

def worker(task_queue, result_queue, gpu_id):
    """Worker process that runs experiments on a specific GPU"""
    while True:
        task = task_queue.get()

        if task is None:  # Poison pill to stop worker
            break

        model, dataset, dataset_type = task

        start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[GPU {gpu_id}] [{start_timestamp}] Starting {model} on {dataset} ({dataset_type})")
        start_time = time.time()

        # Determine script to run
        if dataset_type == 'HAR':
            script_path = f'codes/{model}/run_har_experiments.py'
        else:
            script_path = f'codes/{model}/run_tsc_experiments.py'

        # Create temp script in the SAME directory as the original script
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        temp_script = os.path.join(script_dir, f'temp_{dataset}_{gpu_id}_{script_name}')

        # Read and modify the script
        try:
            with open(script_path, 'r') as f:
                content = f.read()

            # Modify dataset selection
            lines = content.split('\n')
            modified_lines = []
            for line in lines:
                # Comment out all dataset assignments
                if line.strip().startswith('dataset =') and '#' not in line.split('dataset')[0]:
                    if f'"{dataset}"' in line or f"'{dataset}'" in line:
                        modified_lines.append(line)  # Keep target dataset uncommented
                    else:
                        modified_lines.append('#' + line)  # Comment out others
                # Uncomment target dataset if commented
                elif line.strip().startswith('#dataset =') and (f'"{dataset}"' in line or f"'{dataset}'" in line):
                    modified_lines.append(line.lstrip('#').lstrip())
                else:
                    modified_lines.append(line)

            # Write temp script
            with open(temp_script, 'w') as f:
                f.write('\n'.join(modified_lines))

            # Set GPU environment
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            # Run experiment from project root
            result = subprocess.run(
                [sys.executable, temp_script],
                env=env,
                capture_output=True,
                text=True,
                timeout=3600
            )

            elapsed = time.time() - start_time
            success = result.returncode == 0

            print(f"[GPU {gpu_id}] Completed {model} on {dataset} - {'SUCCESS' if success else 'FAILED'} ({elapsed:.1f}s)")

            # Save log
            log_dir = f"logs_{dataset_type.lower()}/{model}"
            os.makedirs(log_dir, exist_ok=True)
            with open(f"{log_dir}/{dataset}_log.txt", 'w') as f:
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
            if 'temp_script' in locals() and os.path.exists(temp_script):
                try:
                    os.remove(temp_script)
                except:
                    pass

def main():
    parser = argparse.ArgumentParser(description='Run experiments in parallel across multiple GPUs')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='GPU IDs to use (e.g., --gpus 0 1 2 3)')
    parser.add_argument('--processes-per-gpu', type=int, default=1,
                        help='Number of processes to run per GPU (default: 1)')
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
    num_workers = num_gpus * args.processes_per_gpu

    print(f"{'='*80}")
    print(f"PARALLEL EXPERIMENT RUNNER")
    print(f"{'='*80}")
    print(f"GPUs: {args.gpus} ({num_gpus} GPUs)")
    print(f"Processes per GPU: {args.processes_per_gpu}")
    print(f"Total workers: {num_workers}")
    print(f"Split strategy: {args.split}")
    print(f"HAR experiments: {args.har}")
    print(f"TSC experiments: {args.tsc}")
    print(f"{'='*80}\n")

    # Create task list
    all_tasks = []

    if args.har:
        for model in MODELS:
            for dataset in HAR_DATASETS:
                all_tasks.append((model, dataset, 'HAR'))

    if args.tsc:
        for model in MODELS:
            for dataset in TSC_DATASETS:
                all_tasks.append((model, dataset, 'TSC'))

    # Filter out already successful experiments
    print("Checking for already completed experiments...")
    tasks = []
    skipped = []
    for task in all_tasks:
        model, dataset, dataset_type = task
        if is_experiment_successful(model, dataset, dataset_type):
            skipped.append(task)
            print(f"  ✓ Skipping {model}/{dataset} ({dataset_type}) - already successful")
        else:
            tasks.append(task)

    print(f"\nTotal experiments: {len(all_tasks)}")
    print(f"Already completed: {len(skipped)}")
    print(f"To run: {len(tasks)}")
    print(f"Estimated time with {num_workers} workers: ~{(len(tasks) / num_workers) * 20} minutes")
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
    for _ in range(num_workers):
        task_queue.put(None)

    # Start workers
    print(f"Starting {num_workers} worker processes...\n")
    workers = []
    for i in range(num_workers):
        gpu_id = args.gpus[i % num_gpus]  # Cycle through GPUs
        p = Process(target=worker, args=(task_queue, result_queue, gpu_id))
        p.start()
        workers.append(p)
        print(f"  Worker {i+1}/{num_workers} assigned to GPU {gpu_id}")

    # Collect results
    results = []
    for _ in range(len(tasks)):
        result = result_queue.get()
        results.append(result)

    # Wait for all workers to finish
    for p in workers:
        p.join()

    # Print summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")

    total_tasks = len(results)
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
