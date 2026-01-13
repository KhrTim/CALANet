# Updated LaTeX Tables - Complete Metrics by Dataset

## ✅ What Was Fixed

### 1. NaN Bug Fixed
All 23 experiments (12 CALANet + 11 millet) were re-run with fixed training metrics tracking:
- **Before**: Training time and peak memory were `null` for CALANet and millet
- **After**: All metrics are now complete with no missing values

### 2. New Table Format
Tables restructured per your request:
- **Rows**: Models (one row per model)
- **Columns**: Datasets (one column per dataset)
- **Each metric has its own separate table**

## 📊 Generated Files

### Main Files
- **`metrics_by_dataset_tables.tex`** - Complete LaTeX document with all tables
- **`metrics_by_dataset_tables.pdf`** - Compiled PDF (72 KB, 16 pages)

### Previous Files (for reference)
- `comprehensive_metrics_tables.tex` - Old format (averaged metrics)
- `detailed_results_tables.tex` - Old format (per-dataset breakdown)

## 📋 Table Organization

The document contains **16 tables total** (8 for HAR + 8 for TSC):

### HAR Tables (6 datasets)
Datasets: UCI_HAR, DSADS, OPPORTUNITY, KU-HAR, PAMAP2, REALDISP

1. **Table 1**: Accuracy (%)
2. **Table 2**: F1-Score Macro (%)
3. **Table 3**: Precision Macro (%)
4. **Table 4**: Recall Macro (%)
5. **Table 5**: Training Time (minutes)
6. **Table 6**: Inference Throughput (samples/sec)
7. **Table 7**: Parameters (millions)
8. **Table 8**: Peak Memory (GB)

### TSC Tables (6 datasets)
Datasets: AtrialFibrillation, Heartbeat, LSST, MotorImagery, PEMS-SF, PhonemeSpectra

9. **Table 9**: Accuracy (%)
10. **Table 10**: F1-Score Macro (%)
11. **Table 11**: Precision Macro (%)
12. **Table 12**: Recall Macro (%)
13. **Table 13**: Training Time (minutes)
14. **Table 14**: Inference Throughput (samples/sec)
15. **Table 15**: Parameters (millions)
16. **Table 16**: Peak Memory (GB)

## 📈 Example Table Format

```latex
\begin{table}[htbp]
\caption{Accuracy (\%) for HAR Models}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{UCI_HAR} & \textbf{DSADS} & ... \\
\midrule
SAGOG          & 28.64            & 3.86           & ... \\
MPTSNet        & 60.60            & 82.98          & ... \\
MSDL           & 79.03            & 86.01          & ... \\
CALANet        & 89.48            & 80.96          & ... \\
millet         & 93.08            & 82.98          & ... \\
...
\bottomrule
\end{tabular}
\end{table}
```

## 🎯 Key Highlights

### Complete Data
- **113/120 experiments** (94.2%) successful
- **Zero NaN values** - all training metrics now complete
- All 14 models included (GTWIDL excluded as requested)

### Training Metrics Now Available
Example from CALANet UCI_HAR:
- Training Time: 737.1s (12.3 minutes) ✅
- Peak Memory: 0.31 GB ✅
- Inference Throughput: 15,941 samples/sec ✅
- Parameters: 0.67M ✅

## 📝 How to Use in Your Paper

### Option 1: Include All Tables
```latex
\documentclass{article}
\usepackage{booktabs}
\begin{document}

% Include the generated tables
\input{metrics_by_dataset_tables}

\end{document}
```

### Option 2: Select Specific Tables
Copy individual tables from `metrics_by_dataset_tables.tex` into your paper:
- Use Accuracy and F1-Score for main results
- Use Training Time and Parameters for efficiency analysis
- Use Peak Memory for resource requirements discussion

### Option 3: Customize Further
The Python script `generate_metric_tables.py` can be modified to:
- Add/remove metrics
- Change formatting (decimal places, units)
- Add statistical significance markers
- Highlight best values per dataset

## 🔧 Regenerating Tables

If you need to regenerate the tables after adding more results:

```bash
python3 generate_metric_tables.py
```

This will:
1. Load all metrics from `results/*/`
2. Generate `metrics_by_dataset_tables.tex`
3. Compile to PDF automatically

## 📊 Data Sources

All data loaded from:
```
results/
├── CALANet/
│   ├── UCI_HAR_metrics.json
│   ├── DSADS_metrics.json
│   └── ...
├── millet/
│   ├── UCI_HAR_metrics.json
│   └── ...
├── SAGOG/
├── MPTSNet/
├── MSDL/
└── ...
```

Each JSON file contains:
- `effectiveness` metrics: accuracy, f1_macro, precision_macro, recall_macro
- `efficiency` metrics: training_time_seconds, inference_throughput, total_parameters, peak_memory_gb

## ✨ Summary

**What changed:**
1. ✅ Fixed NaN bug in CALANet and millet (re-ran 23 experiments)
2. ✅ All 113 experiments now have complete metrics
3. ✅ Restructured tables: models as rows, datasets as columns
4. ✅ Each metric has its own dedicated table
5. ✅ Generated both .tex and .pdf files ready for publication

**Result:**
Your experiment results are now complete and presentation-ready for your paper!
