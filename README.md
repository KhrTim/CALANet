# CALANet: Channel-Aware Lightweight Attention Network for HAR and TSC

Reproducible implementation of CALANet and baseline models for Human Activity Recognition (HAR) and Time Series Classification (TSC).

## Repository Structure

```
├── codes/                    # Model implementations
│   ├── CALANet_local/       # Proposed model (CALANet)
│   ├── Bi-GRU-I/            # Bidirectional GRU baseline
│   ├── DeepConvLSTM/        # DeepConvLSTM baseline
│   ├── DSN-master/          # DSN baseline
│   ├── FCN_TSC/             # Fully Convolutional Network (TSC)
│   ├── IF-ConvTransformer2/ # IF-ConvTransformer baseline
│   ├── InceptionTime/       # InceptionTime baseline (TSC)
│   ├── millet/              # MILLET baseline
│   ├── MPTSNet/             # MPTSNet baseline
│   ├── MSDL/                # MSDL baseline
│   ├── RepHAR/              # RepHAR baseline
│   ├── resnet/              # ResNet baseline (TSC)
│   ├── RevTransformerAttentionHAR/  # RevAttNet baseline
│   ├── SAGOG/               # SAGoG baseline
│   └── shared_metrics.py    # Shared metrics collection
├── Data/                    # Datasets (HAR & TSC)
├── results/                 # Experiment results (JSON metrics)
├── run_all_har_experiments.py    # Run all HAR experiments
├── run_all_tsc_experiments.py    # Run all TSC experiments
└── run_parallel_experiments.py   # Parallel experiment runner
```

## Datasets

**HAR (6 datasets):** UCI-HAR, DSADS, OPPORTUNITY, KU-HAR, PAMAP2, REALDISP

**TSC (6 datasets):** AtrialFibrillation, MotorImagery, Heartbeat, PhonemeSpectra, LSST, PEMS-SF

## Running Experiments

### Single Model
```bash
cd codes/CALANet_local
python run.py  # For HAR datasets
python run_TSC.py  # For TSC datasets
```

### All Experiments
```bash
python run_all_har_experiments.py
python run_all_tsc_experiments.py
```

### Parallel Execution
```bash
python run_parallel_experiments.py
```

## Results

Results are saved in `results/{model_name}/{dataset}_metrics.json` containing:
- Accuracy, F1-score, Precision, Recall
- Training time, Inference throughput
- Memory usage, Parameter count
- Per-class metrics and confusion matrices

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy, scikit-learn

## Citation

If you use this code, please cite our paper.
