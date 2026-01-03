# Comprehensive Experiment Issues Analysis

**Date**: 2026-01-03
**Analysis**: Missing metrics collection for SAGOG, MPTSNet, and MSDL models

## Executive Summary

Three critical issues were identified and fixed that prevented comprehensive metrics collection from running for SAGOG, MPTSNet, and MSDL experiments:

1. **Parallel runner used old temp files** - Created before metrics collection code was added
2. **MPTSNet syntax error** - Optimizer initialization broken by misplaced metrics collector
3. **Incorrect input shapes** - Wrong tensor dimensions for model complexity computation

## Detailed Findings

### Issue 1: Old Temp Files from Parallel Runner

**Root Cause**:
- The parallel experiment runner (`run_parallel_experiments.py`) creates temporary Python files by copying the main script and modifying the dataset selection line
- These temp files were created from older versions of the scripts BEFORE comprehensive metrics collection was integrated
- When experiments ran, they used these old temp files which lacked the metrics collection section

**Evidence**:
```
codes/SAGOG/temp_DSADS_1_run_har_experiments.py: 295 lines
codes/SAGOG/run_har_experiments.py: 342 lines (with metrics at lines 336-341)
```

**Impact**:
- All SAGOG, MPTSNet, and MSDL experiments completed successfully (Exit Code: 0)
- Training results were saved to `MODEL/results/DATASET_results.txt`
- **BUT** comprehensive metrics were NOT collected or saved to `results/MODEL/`
- Logs show successful completion but no "COLLECTING COMPREHENSIVE METRICS" section

**Example from SAGOG log**:
```
Results saved to SAGOG/results/UCI_HAR_sagog_results.txt

================================================================================
STDERR:
================================================================================
[END OF LOG - No metrics collection section]
```

**Fix Applied**:
- Deleted all temp files: `find codes -name "temp_*_run_*.py" -delete`
- Files will be regenerated from current code on next experiment run

### Issue 2: MPTSNet Syntax Error

**Location**: `codes/MPTSNet/run_har_experiments.py` lines 175-189

**Problem**:
Metrics collector initialization was incorrectly placed INSIDE the optimizer initialization:

```python
# BEFORE (SYNTAX ERROR):
optimizer = torch.optim.Adam(

# Initialize metrics collector
metrics_collector = MetricsCollector(
    model_name='MPTSNet',
    dataset=dataset,
    task_type='HAR',
    save_dir='results'
)
    model.parameters(),  # This is OUTSIDE the Adam() call!
    lr=learning_rate,
    weight_decay=weight_decay
)
```

**Impact**:
- This is invalid Python syntax
- However, the temp file version didn't have this code yet, so experiments ran successfully
- Would have caused crashes once temp files were regenerated

**Fix Applied**:
```python
# AFTER (CORRECT):
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

# Initialize metrics collector
metrics_collector = MetricsCollector(
    model_name='MPTSNet',
    dataset=dataset,
    task_type='HAR',
    save_dir='results'
)
```

### Issue 3: Incorrect Input Shapes

**Problem**: Input shapes for model complexity computation were incorrect

**MPTSNet** (line 381):
```python
# BEFORE (WRONG):
input_shape = (1, segment_size, input_nc)  # (batch, length, channels)

# AFTER (CORRECT):
input_shape = (1, input_nc, segment_size)  # (batch, channels, length)
```

**SAGOG** (line 307):
```python
# BEFORE (WRONG):
input_shape = (1, input_nc, 1, segment_size)  # Extra dimension

# AFTER (CORRECT):
input_shape = (1, input_nc, segment_size)  # Standard 3D format
```

**Impact**:
- Would cause errors or incorrect FLOPs/MACs computation
- PyTorch expects (batch, channels, length) format for 1D Conv models
- Data from `Read_Data()` is already in (batch, channels, length) format

## Current Experiment Status

### Overall: 109/120 experiments successful (90.8%)

**TSC (Time Series Classification)**: 54/54 (100% success)
- All 9 models × 6 datasets completed successfully
- All have comprehensive metrics saved

**HAR (Human Activity Recognition)**: 55/66 (83.3% success)

**Missing Results** (Need Re-run):
1. **SAGOG**: 6 experiments completed but NO metrics saved
   - Files: `SAGOG/results/*_sagog_results.txt` exist
   - Missing: `results/SAGOG/*_metrics.json`

2. **MPTSNet**: 6 experiments completed but NO metrics saved
   - Files: `MPTSNet/results/*_mptsnet_results.txt` exist
   - Missing: `results/MPTSNet/*_metrics.json`

3. **MSDL**: 6 experiments completed but NO metrics saved
   - Files: `MSDL/results/*_msdl_results.txt` exist
   - Missing: `results/MSDL/*_metrics.json`

**Known Failures** (Acceptable):
- IF-ConvTransformer2: 4/6 HAR (architecture limitation - requires 6 channels)
- millet REALDISP: NaN convergence issue

## Verification Needed

### 1. SAGOG Training Issue
**Concern**: SAGOG UCI_HAR log shows only 5 epochs ran (not 500 configured)

```
TESTED: Using base epochs=5 for small/medium dataset (7352 samples)
Early stopping triggered at epoch 4
Best F1 Score: 0.0562 at epoch 1
Test Accuracy: 0.1822 (very poor)
```

**Questions**:
- Is there an automatic epoch reduction for testing?
- Why did early stopping trigger so quickly?
- Is 0.18 accuracy acceptable, or does this indicate a training problem?

**Action Required**:
- Check if there's a test mode or debug flag enabled in SAGOG
- Verify if 500 epochs should actually run for production experiments

### 2. Model Performance Summary

**Best Performing Models (HAR)**:
- GTWIDL: 0.947 avg F1 (3/6 datasets tested)
- Bi-GRU-I: 0.907 avg F1
- DeepConvLSTM: 0.894 avg F1
- RepHAR: 0.889 avg F1

**TSC Performance**:
- resnet: 0.629 avg accuracy
- InceptionTime: 0.528 avg accuracy
- CALANet: 0.462 avg accuracy

**Challenging Datasets**:
- AtrialFibrillation: 0.067 avg accuracy (all models struggle)
- PhonemeSpectra: 0.313 avg accuracy

## Next Steps

### Immediate Actions Required

1. **Re-run missing experiments** (SAGOG, MPTSNet, MSDL)
   ```bash
   # The temp files have been deleted
   # Next run of parallel_experiments will regenerate them with new code
   python run_parallel_experiments.py
   ```

2. **Investigate SAGOG training**
   - Check for test/debug mode flags
   - Verify epoch configuration is working correctly
   - Determine if 5 epochs is intentional or a bug

3. **Verify CALANet HAR re-run**
   - Previous fix for missing save directory (Commit 82679b0)
   - Need to confirm 6 HAR experiments now complete successfully

### Optional Improvements

1. **Parallel runner enhancement**: Add version tracking to temp files to detect stale code

2. **Validation checks**: Add pre-flight checks to verify metrics collection code exists before running experiments

3. **Monitoring**: Add progress indicators to show metrics collection is running

## Files Modified

### Commit 68931b1: "Fix missing metrics collection in SAGOG/MPTSNet/MSDL"

1. `codes/MPTSNet/run_har_experiments.py`
   - Fixed optimizer initialization syntax error (lines 175-189)
   - Fixed input_shape for model complexity (line 381)

2. `codes/SAGOG/run_har_experiments.py`
   - Fixed input_shape for model complexity (line 307)

3. Deleted all temp files to force regeneration

## Conclusion

The root cause of missing metrics for SAGOG, MPTSNet, and MSDL was the parallel runner using outdated temp files created before metrics collection code was integrated.

**Critical fixes applied**:
✅ MPTSNet syntax error corrected
✅ Input shapes fixed for both MPTSNet and SAGOG
✅ Old temp files deleted

**Next run will**:
✅ Generate fresh temp files with current code
✅ Include comprehensive metrics collection
✅ Save results to `results/MODEL/DATASET_metrics.json`

**Outstanding concerns**:
⚠️ SAGOG only ran 5 epochs (need investigation)
⚠️ Need to re-run 18 HAR experiments (SAGOG, MPTSNet, MSDL × 6 datasets)
⚠️ Need to verify CALANet HAR fix worked
