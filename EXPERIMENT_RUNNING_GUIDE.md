# Experiment Running Guide

Complete guide for running experiments with comprehensive metrics collection.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Single Experiment](#single-experiment)
- [Batch Experiments](#batch-experiments)
- [Parallel Experiments (Recommended)](#parallel-experiments-recommended)
- [Results Location](#results-location)
- [Monitoring Progress](#monitoring-progress)

---

## Prerequisites

**Python Environment:**
The experiments require PyTorch 2.0+ with CUDA support. If you're using conda:

```bash
# Activate the rthar environment (or your PyTorch environment)
conda activate rthar

# Verify PyTorch is available
python -c "import torch; print(f'PyTorch {torch.__version__} CUDA: {torch.cuda.is_available()}')"
```

**Note:** The parallel runner automatically uses `/userHome/userhome1/timur/miniconda3/envs/rthar/bin/python` if available.

---

## Quick Start

### Running a Single Model on One Dataset

```bash
# Example: Run SAGOG on UCI_HAR dataset
cd codes/SAGOG
python run_har_experiments.py  # Uses default dataset in the script

# Or run directly from project root
python codes/SAGOG/run_har_experiments.py
```

### Running All Experiments (Parallel - RECOMMENDED)

```bash
# Use all available GPUs
python run_parallel_experiments.py --gpus 0,1,2,3

# Use specific GPUs
python run_parallel_experiments.py --gpus 0,1

# Run only HAR experiments
python run_parallel_experiments.py --gpus 0,1 --task-type HAR

# Run only TSC experiments
python run_parallel_experiments.py --gpus 0,1 --task-type TSC
```

---

## Single Experiment

### Step 1: Navigate to Model Directory

```bash
cd codes/<MODEL_NAME>
```

### Step 2: Edit the Script to Select Dataset

Open the experiment script and uncomment the dataset you want:

**For HAR experiments** (`run_har_experiments.py`):
```python
# Dataset selection (uncomment the one you want to run)
#dataset = "UCI_HAR"
dataset = "DSADS"  # <- This one will run
#dataset = "OPPORTUNITY"
#dataset = "KU-HAR"
#dataset = "PAMAP2"
#dataset = "REALDISP"
```

**For TSC experiments** (`run_tsc_experiments.py`):
```python
# Dataset selection (uncomment the one you want to run)
#dataset = "AtrialFibrillation"
dataset = "MotorImagery"  # <- This one will run
#dataset = "Heartbeat"
#dataset = "PhonemeSpectra"
#dataset = "LSST"
#dataset = "PEMS-SF"
```

### Step 3: Run the Script

```bash
# Run with default GPU (GPU 0)
python run_har_experiments.py

# Or specify GPU
CUDA_VISIBLE_DEVICES=1 python run_har_experiments.py
```

### Models Available

#### HAR (Human Activity Recognition):
- **SAGOG**: `codes/SAGOG/run_har_experiments.py`
- **GTWIDL**: `codes/GTWIDL/run_har_experiments.py`
- **MPTSNet**: `codes/MPTSNet/run_har_experiments.py`
- **MSDL**: `codes/MSDL/run_har_experiments.py`
- **RepHAR**: `codes/RepHAR/run.py`
- **DeepConvLSTM**: `codes/DeepConvLSTM/run.py`
- **Bi-GRU-I**: `codes/Bi-GRU-I/run.py`
- **RevTransformerAttentionHAR**: `codes/RevTransformerAttentionHAR/run.py`
- **IF-ConvTransformer2**: `codes/IF-ConvTransformer2/run.py`
- **millet**: `codes/millet/run.py`
- **DSN-master**: `codes/DSN-master/run.py`

#### TSC (Time Series Classification):
- **SAGOG**: `codes/SAGOG/run_tsc_experiments.py`
- **GTWIDL**: `codes/GTWIDL/run_tsc_experiments.py`
- **MPTSNet**: `codes/MPTSNet/run_tsc_experiments.py`
- **MSDL**: `codes/MSDL/run_tsc_experiments.py`
- **millet**: `codes/millet/run_TSC.py`
- **DSN-master**: `codes/DSN-master/run_TSC.py`
- **resnet**: `codes/resnet/run_TSC.py`
- **FCN_TSC**: `codes/FCN_TSC/run_TSC.py`
- **InceptionTime**: `codes/InceptionTime/run_TSC.py`

---

## Batch Experiments

### Run All HAR Experiments Sequentially

```bash
python run_all_har_experiments.py --gpu 0
```

This runs:
- All 4 main models (SAGOG, GTWIDL, MPTSNet, MSDL)
- On all 6 HAR datasets (UCI_HAR, DSADS, OPPORTUNITY, KU-HAR, PAMAP2, REALDISP)
- Total: 24 experiments sequentially

### Run All TSC Experiments Sequentially

```bash
python run_all_tsc_experiments.py --gpu 0
```

This runs:
- All 4 main models (SAGOG, GTWIDL, MPTSNet, MSDL)
- On all 6 TSC datasets (AtrialFibrillation, MotorImagery, Heartbeat, PhonemeSpectra, LSST, PEMS-SF)
- Total: 24 experiments sequentially

---

## Parallel Experiments (RECOMMENDED)

The parallel runner distributes experiments across multiple GPUs for maximum efficiency.

**Now includes ALL 15 models:**
- 4 main models: SAGOG, GTWIDL, MPTSNet, MSDL
- 5 HAR-only: RepHAR, DeepConvLSTM, Bi-GRU-I, RevTransformerAttentionHAR, IF-ConvTransformer2
- 2 both HAR+TSC: millet, DSN
- 3 TSC-only: resnet, FCN, InceptionTime

### Basic Usage

```bash
# Run ALL experiments (HAR + TSC) on GPUs 0, 1, 2, 3
python run_parallel_experiments.py --gpus 0 1 2 3

# Run only HAR experiments
python run_parallel_experiments.py --gpus 0 1 --har

# Run only TSC experiments
python run_parallel_experiments.py --gpus 0 1 --tsc
```

### Advanced Options

```bash
# Use more GPUs for faster execution
python run_parallel_experiments.py --gpus 0 1 2 3 4 5 6 7

# Split work by dataset (run all models on same dataset together)
python run_parallel_experiments.py --gpus 0 1 --split dataset --har

# Split work by model (run same model on all datasets together)
python run_parallel_experiments.py --gpus 0 1 --split model --tsc

# Run multiple processes per GPU (if GPU has enough memory)
python run_parallel_experiments.py --gpus 0 1 --processes-per-gpu 2
```

### How It Works

1. Creates a queue of all experiment combinations (model × dataset)
2. Spawns worker processes, one per GPU
3. Each worker picks experiments from the queue
4. Automatically sets `CUDA_VISIBLE_DEVICES` for each worker
5. Logs all output to `logs_har/<MODEL>/<DATASET>_log.txt` or `logs_tsc/<MODEL>/<DATASET>_log.txt`

### Example Output

```
[GPU 0] [2025-12-16 14:30:15] Starting SAGOG on UCI_HAR (HAR)
[GPU 1] [2025-12-16 14:30:15] Starting GTWIDL on DSADS (HAR)
[GPU 2] [2025-12-16 14:30:15] Starting MPTSNet on OPPORTUNITY (HAR)
[GPU 3] [2025-12-16 14:30:15] Starting MSDL on KU-HAR (HAR)

[GPU 0] [2025-12-16 14:45:23] SAGOG on UCI_HAR completed (15m 8s) ✓
[GPU 0] [2025-12-16 14:45:24] Starting SAGOG on PAMAP2 (HAR)
...
```

---

## Results Location

### Metrics Files (NEW - Comprehensive)

All experiments now save comprehensive metrics:

```
results/
├── <MODEL>_<DATASET>_<TASK>_metrics.json      # Machine-readable
└── <MODEL>_<DATASET>_<TASK>_metrics.txt       # Human-readable summary
```

**Example:**
```
results/SAGOG_UCI_HAR_HAR_metrics.json
results/SAGOG_UCI_HAR_HAR_metrics.txt
```

**Metrics Included:**
- **Effectiveness**: Accuracy, Precision, Recall, F1-score (weighted/macro), Confusion Matrix
- **Efficiency**: Training time, inference time, throughput (samples/sec)
- **Model Complexity**: Parameters, FLOPs, peak GPU memory usage

### Model-Specific Results

Some models also save to their own directories:
```
codes/SAGOG/results/
codes/GTWIDL/results/
codes/MPTSNet/results/
codes/MSDL/results/
...
```

### Execution Logs

All experiment output is logged:
```
logs_har/<MODEL>/<DATASET>_log.txt
logs_tsc/<MODEL>/<DATASET>_log.txt
```

### Saved Models

Best models are saved during training:
```
codes/<MODEL>/save/<DATASET>_<model>.pt
```

---

## Monitoring Progress

### Check Running Experiments

```bash
# See GPU usage
nvidia-smi

# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Monitor log files in real-time
tail -f logs_har/SAGOG/UCI_HAR_log.txt
```

### Check Completed Experiments

```bash
# List all completed HAR experiments
find logs_har -name "*_log.txt" -exec grep -l "Exit Code: 0" {} \;

# List all completed TSC experiments
find logs_tsc -name "*_log.txt" -exec grep -l "Exit Code: 0" {} \;

# Count completed experiments
find logs_har logs_tsc -name "*_log.txt" -exec grep -l "Exit Code: 0" {} \; | wc -l
```

### Check Metrics

```bash
# View human-readable metrics
cat results/SAGOG_UCI_HAR_HAR_metrics.txt

# View JSON metrics
python -m json.tool results/SAGOG_UCI_HAR_HAR_metrics.json

# List all collected metrics
ls -lh results/*_metrics.json
```

---

## Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size in the experiment script
# Edit codes/<MODEL>/run_*_experiments.py
batch_size = 64  # Try reducing to 32 or 16
```

### CUDA Error

```bash
# Clear GPU memory
pkill -9 python

# Verify GPU availability
nvidia-smi

# Check CUDA setup
python -c "import torch; print(torch.cuda.is_available())"
```

### Experiment Hangs

```bash
# Kill specific experiment
pkill -f "run_har_experiments.py"

# Check for stuck processes
ps aux | grep python
```

### Re-run Failed Experiments

```bash
# The parallel runner can skip completed experiments
python run_parallel_experiments.py --gpus 0,1 --skip-completed

# Or manually delete the log file to force re-run
rm logs_har/SAGOG/UCI_HAR_log.txt
```

---

## Best Practices

1. **Use Parallel Runner**: Much faster than sequential execution
   ```bash
   python run_parallel_experiments.py --gpus 0,1,2,3
   ```

2. **Start with Small Tests**: Test one model/dataset first
   ```bash
   cd codes/SAGOG
   python run_har_experiments.py  # Edit to select UCI_HAR
   ```

3. **Monitor Resources**: Keep an eye on GPU memory
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Check Logs**: Always verify experiments completed successfully
   ```bash
   grep "Exit Code" logs_har/SAGOG/UCI_HAR_log.txt
   ```

5. **Backup Results**: Copy results periodically
   ```bash
   tar -czf results_backup_$(date +%Y%m%d).tar.gz results/
   ```

---

## Expected Runtime

**Approximate times per experiment** (on modern GPU):

| Model     | HAR Dataset | TSC Dataset |
|-----------|-------------|-------------|
| SAGOG     | 10-20 min   | 15-30 min   |
| GTWIDL    | 5-15 min    | 10-20 min   |
| MPTSNet   | 15-30 min   | 20-40 min   |
| MSDL      | 15-30 min   | 20-40 min   |

**Total time estimates (with ALL models):**
- All HAR experiments (11 models × 6 datasets = 66): ~18-35 hours sequentially
- All TSC experiments (9 models × 6 datasets = 54): ~15-30 hours sequentially
- **Grand total**: 120 experiments
- **With 4 GPUs in parallel**: ~5-10 hours for everything!
- **With 8 GPUs in parallel**: ~3-5 hours for everything!

---

## Questions?

If you encounter issues:
1. Check the experiment log: `logs_har/<MODEL>/<DATASET>_log.txt`
2. Verify the dataset exists: `ls Data/HAR/<DATASET>/` or `ls Data/TSC/<DATASET>/`
3. Check GPU availability: `nvidia-smi`
4. Review the metrics output: `cat results/<MODEL>_<DATASET>_<TASK>_metrics.txt`
