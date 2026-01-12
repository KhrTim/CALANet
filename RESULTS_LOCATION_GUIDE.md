# Complete Results Location Guide

## 📊 All Your Results Are Here!

### 1. **Comprehensive Metrics - Individual JSON Files**
**Location**: `results/{MODEL}/{DATASET}_metrics.json`

**Example**:
```
results/SAGOG/UCI_HAR_metrics.json
results/MPTSNet/DSADS_metrics.json
results/MSDL/AtrialFibrillation_metrics.json
```

**What's inside each JSON file**:
```json
{
  "model": "SAGOG",
  "dataset": "UCI_HAR",
  "task_type": "HAR",
  "timestamp": "2026-01-03T...",

  "effectiveness": {
    "accuracy": 0.2864,
    "precision_macro": 0.2578,
    "precision_weighted": 0.2478,
    "precision_micro": 0.2864,
    "recall_macro": 0.2807,
    "recall_weighted": 0.2864,
    "recall_micro": 0.2864,
    "f1_macro": 0.2041,
    "f1_weighted": 0.2003,
    "f1_micro": 0.2864
  },

  "efficiency": {
    "training_time_seconds": 8301.09,
    "training_time_minutes": 138.35,
    "peak_memory_allocated_gb": 0.835,
    "peak_memory_reserved_gb": 0.857,
    "inference_time_seconds": 20.25,
    "inference_throughput_samples_per_sec": 145.56,
    "total_parameters": 196743,
    "trainable_parameters": 196743,
    "parameters_millions": 0.197,
    "avg_epoch_time_seconds": 61.27,
    "total_epochs": 102
  },

  "training_history": {
    "loss": [...],
    "accuracy": [...]
  }
}
```

**Total**: 116 JSON files (one per successful experiment)

---

### 2. **Combined CSV - All Metrics**
**Location**: `ALL_EXPERIMENTS_RESULTS.csv`

**Contents**:
- All 116 experiments in one file
- 30 metric columns
- Easy to open in Excel, Google Sheets, or Python/R

**Columns include**:
- Model, Dataset, Task Type
- All effectiveness metrics (accuracy, precision, recall, F1)
- All efficiency metrics (time, memory, parameters, throughput)
- Timestamp

**How to use**:
```python
import pandas as pd
df = pd.read_csv('ALL_EXPERIMENTS_RESULTS.csv')
print(df.head())
```

---

### 3. **Summary CSV - Key Metrics Only**
**Location**: `SUMMARY_RESULTS.csv`

**Contents**:
- 116 experiments with simplified view
- 10 key columns for quick analysis

**Columns**:
1. Model
2. Dataset
3. Task (HAR/TSC)
4. Accuracy
5. F1_Macro
6. F1_Weighted
7. Train_Time_Min
8. Inference_TPS (samples per second)
9. Params_M (millions)
10. Memory_GB

**Perfect for**:
- Quick comparisons
- Creating plots
- Statistical analysis

---

### 4. **Execution Logs**
**Location**: `logs_har/{MODEL}/{DATASET}_log.txt` and `logs_tsc/{MODEL}/{DATASET}_log.txt`

**Example**:
```
logs_har/SAGOG/UCI_HAR_log.txt
logs_tsc/MPTSNet/AtrialFibrillation_log.txt
```

**What's inside**:
- Command executed
- GPU used
- Duration
- Exit code
- Full training output (all epochs)
- Classification reports
- Any errors (in STDERR section)

**Total**: 132 log files (120 experiments + 12 old ones)

---

### 5. **Basic Results (Old Format)**
**Location**: `{MODEL}/results/{DATASET}_{model}_results.txt`

**Example**:
```
SAGOG/results/UCI_HAR_sagog_results.txt
MPTSNet/results/DSADS_mptsnet_results.txt
```

**Contents**: Simple text file with accuracy and F1 scores (legacy format)

---

## 📈 How to Analyze Your Results

### Quick Analysis in Python
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load all results
df = pd.read_csv('ALL_EXPERIMENTS_RESULTS.csv')

# Filter HAR experiments
har = df[df['task_type'] == 'HAR']

# Compare models by accuracy
model_acc = har.groupby('model')['effectiveness_accuracy'].mean().sort_values()
print(model_acc)

# Plot training time vs accuracy
plt.scatter(har['efficiency_training_time_minutes'],
            har['effectiveness_accuracy'])
plt.xlabel('Training Time (minutes)')
plt.ylabel('Accuracy')
plt.title('Training Time vs Accuracy (HAR)')
plt.show()
```

### Load Individual JSON
```python
import json

with open('results/SAGOG/UCI_HAR_metrics.json', 'r') as f:
    data = json.load(f)

print(f"Accuracy: {data['effectiveness']['accuracy']}")
print(f"F1 Score: {data['effectiveness']['f1_macro']}")
print(f"Training Time: {data['efficiency']['training_time_minutes']} min")
print(f"Parameters: {data['efficiency']['parameters_millions']}M")
```

### Excel/Google Sheets
1. Open `ALL_EXPERIMENTS_RESULTS.csv`
2. Use pivot tables to compare models
3. Create charts for visualization

---

## 🎯 Quick Stats

**Experiments**: 116/120 successful (94.2%)
- **HAR**: 61/66 (92.4%)
- **TSC**: 52/54 (96.3%)

**Models**: 20 total
- 11 HAR models
- 9 TSC models

**Datasets**: 12 total
- 6 HAR datasets
- 6 TSC datasets

**Metrics Collected Per Experiment**:
- ✅ 10 effectiveness metrics (accuracy, precision, recall, F1)
- ✅ 10+ efficiency metrics (time, memory, throughput, parameters)
- ✅ Full training history
- ✅ Model complexity (FLOPs/MACs when available)

---

## 📁 File Structure Summary

```
/userHome/userhome1/timur/RTHAR_clean/
│
├── results/                          # Individual JSON metrics
│   ├── SAGOG/
│   │   ├── UCI_HAR_metrics.json
│   │   ├── DSADS_metrics.json
│   │   └── ...
│   ├── MPTSNet/
│   ├── MSDL/
│   └── ...
│
├── logs_har/                         # HAR execution logs
│   ├── SAGOG/
│   │   ├── UCI_HAR_log.txt
│   │   └── ...
│   └── ...
│
├── logs_tsc/                         # TSC execution logs
│   └── ...
│
├── ALL_EXPERIMENTS_RESULTS.csv       # ⭐ All metrics combined
├── SUMMARY_RESULTS.csv               # ⭐ Key metrics only
├── FINAL_STATUS_NO_GTWIDL.md         # Final status report
└── RESULTS_LOCATION_GUIDE.md         # This file
```

---

## 🎉 You're All Set!

All your results are organized and ready for analysis. You have:
- ✅ Individual JSON files for detailed analysis
- ✅ Combined CSV for bulk analysis
- ✅ Summary CSV for quick comparisons
- ✅ Full logs for debugging/verification

**Recommended**: Start with `SUMMARY_RESULTS.csv` for overview, then dive into individual JSON files or `ALL_EXPERIMENTS_RESULTS.csv` for detailed analysis!
