# Experiment Execution Guide

This guide explains how to run all experiments for comparing SAGOG, GTWIDL, MPTSNet, and MSDL across HAR and TSC datasets.

## Overview

- **Models**: SAGOG, GTWIDL, MPTSNet, MSDL
- **HAR Datasets**: UCI-HAR, DSADS, OPPORTUNITY, KU-HAR, PAMAP2, REALDISP (6 datasets)
- **TSC Datasets**: AtrialFibrillation, MotorImagery, Heartbeat, PhonemeSpectra, LSST, PEMS-SF (6 datasets)
- **Total Experiments**: 48 (4 models × 12 datasets)
- **Metrics**:
  - HAR: F1-score, FLOPs
  - TSC: Accuracy, FLOPs

## Quick Start

### Single GPU - Run All Experiments

```bash
# Run all HAR experiments on GPU 0
python run_all_har_experiments.py --gpu 0

# Run all TSC experiments on GPU 0
python run_all_tsc_experiments.py --gpu 0
```

### Multi-GPU - Parallel Execution (Recommended)

For faster execution on a multi-GPU workstation (e.g., 3-4 RTX 3090s):

```bash
# Run ALL experiments in parallel across 4 GPUs
python run_parallel_experiments.py --gpus 0 1 2 3

# Run only HAR experiments across 3 GPUs
python run_parallel_experiments.py --gpus 0 1 2 --har

# Run only TSC experiments across 3 GPUs
python run_parallel_experiments.py --gpus 0 1 2 --tsc
```

### Calculate FLOPs

```bash
# Calculate FLOPs for all models on all datasets
python calculate_flops.py
```

## Detailed Usage

### 1. Individual Model Scripts

Each model has HAR and TSC experiment scripts:

**HAR Experiments:**
- `codes/SAGOG/run_har_experiments.py`
- `codes/GTWIDL/run_har_experiments.py`
- `codes/MPTSNet/run_har_experiments.py`
- `codes/MSDL/run_har_experiments.py`

**TSC Experiments:**
- `codes/SAGOG/run_tsc_experiments.py`
- `codes/GTWIDL/run_tsc_experiments.py`
- `codes/MPTSNet/run_tsc_experiments.py`
- `codes/MSDL/run_tsc_experiments.py`

To run a single model on a single dataset, edit the dataset selection in the script and run:
```bash
cd codes/SAGOG
python run_har_experiments.py
```

### 2. Master Scripts

**Run specific models or datasets:**

```bash
# Run only SAGOG and MPTSNet on all HAR datasets
python run_all_har_experiments.py --models SAGOG MPTSNet --gpu 0

# Run all models on only UCI_HAR and DSADS
python run_all_har_experiments.py --datasets UCI_HAR DSADS --gpu 0

# Run MSDL on Heartbeat (TSC)
python run_all_tsc_experiments.py --models MSDL --datasets Heartbeat --gpu 0
```

### 3. Parallel Execution Options

The parallel runner supports two splitting strategies:

**By Dataset (default):**
```bash
# All models for dataset A on GPU 0, all models for dataset B on GPU 1, etc.
python run_parallel_experiments.py --gpus 0 1 2 3 --split dataset
```

**By Model:**
```bash
# All datasets for SAGOG on GPU 0, all datasets for GTWIDL on GPU 1, etc.
python run_parallel_experiments.py --gpus 0 1 2 3 --split model
```

## Output Structure

### Results

Each model saves results to its own directory:

```
codes/SAGOG/results/{dataset}_sagog_results.txt
codes/GTWIDL/results/{dataset}_gtwidl_results.txt
codes/MPTSNet/results/{dataset}_mptsnet_results.txt
codes/MSDL/results/{dataset}_msdl_results.txt
```

### Logs

```
logs_har/{MODEL}/{DATASET}_log.txt
logs_tsc/{MODEL}/{DATASET}_log.txt
```

### Summaries

```
har_experiments_summary.txt
tsc_experiments_summary.txt
parallel_experiments_summary.txt
flops_results.txt
```

## Expected Runtime

On a single RTX 3090:
- HAR experiment: ~15-30 minutes per model per dataset
- TSC experiment: ~10-20 minutes per model per dataset
- Total (48 experiments): ~16-24 hours

On 4 RTX 3090s in parallel:
- Total (48 experiments): ~4-6 hours

## Troubleshooting

### OOM Errors

SAGOG may experience OOM on high-channel datasets. The scripts use adaptive configurations:
- Batch size: 16 (reduced from 64)
- Hidden dims scaled based on channel count
- Fewer windows for high-channel datasets

### Timeout

Default timeout is 1 hour per experiment. Edit `timeout=3600` in the master scripts if needed.

### Missing Dependencies

```bash
pip install torch torch-geometric aeon thop scikit-learn numpy pandas tqdm
```

## Implementation Notes

### SAGOG
- **Achieved**: 0.722 accuracy on Heartbeat (target: 0.761)
- **Issues**: Class imbalance, memory constraints on high-channel datasets
- **Optimizations**: Reduced architecture for 50+ channels

### GTWIDL
- Uses dictionary learning with time warping
- SVM classifier on learned features
- Slower training due to dictionary optimization

### MPTSNet
- FFT-based period detection
- Adaptive embedding dimensions
- Fast inference

### MSDL
- Multiscale temporal dynamics
- LSTM-based feature extraction
- Configurable kernel sizes

## Next Steps

After running all experiments:

1. **Collect Results**: Results are saved in `codes/{MODEL}/results/`
2. **Calculate FLOPs**: Run `python calculate_flops.py`
3. **Generate Report**: Use the results and FLOPs to create comparison tables
4. **Analyze**: Compare F1/Accuracy vs FLOPs trade-offs

## Contact

For issues or questions, check the implementation in:
- `codes/SAGOG/sagog_model.py`
- `codes/GTWIDL/gtwidl.py`
- `codes/MPTSNet/model/MPTSNet.py`
- `codes/MSDL/msdl.py`
