# Investigation: Discrepancies Between Paper and Collected Results

## Summary

I've identified **CRITICAL ISSUES** with the collected results that explain why they differ from your paper.

---

## Issue 1: **NOT LOADING BEST CHECKPOINT** ❌ (MAJOR BUG)

### Problem
The training code saves the best model checkpoint during training, but **metrics collection evaluates the final epoch model instead of the best checkpoint**.

### Evidence (CALANet UCI-HAR)
```
Paper F1:           96.1%
Best during training: 92.7% (saved to checkpoint)
Collected metric:   89.5% (final epoch - WRONG!)
```

### Impact on All Datasets

| Dataset | Paper F1 | Best Checkpoint | Collected | Gap (Not Loading Best) |
|---------|----------|-----------------|-----------|------------------------|
| UCI_HAR | 96.1% | 92.7% | 89.5% | **-3.2%** |
| DSADS | 90.0% | 86.3% | 79.4% | **-6.8%** |
| OPPORTUNITY | 81.6% | 76.9% | 74.5% | **-2.4%** |
| KU-HAR | 97.5% | 92.6% | 88.4% | **-4.2%** |
| PAMAP2 | 79.4% | 76.7% | 65.3% | **-11.3%** |

### Code Location
`codes/CALANet_local/run.py` lines 157-159:
```python
if max_f1 < weighted_avg_f1:
    os.makedirs('HT-AggNet_v2/save/with_gts', exist_ok=True)
    torch.save(model.state_dict(), 'HT-AggNet_v2/save/with_gts/'+dataset + memo + '.pt')
    # ✓ Best model saved here
```

Then line 182:
```python
with metrics_collector.track_inference():
    eval_loss, y_pred = infer(eval_queue, model, criterion)
    # ❌ Evaluating current model (final epoch), NOT best checkpoint!
```

### Fix Required
Load the best checkpoint before metrics collection:
```python
# After training loop ends
model.load_state_dict(torch.load('HT-AggNet_v2/save/with_gts/'+dataset + memo + '.pt'))

# Then collect metrics
with metrics_collector.track_inference():
    eval_loss, y_pred = infer(eval_queue, model, criterion)
```

---

## Issue 2: **PAPER RESULTS STILL HIGHER THAN OUR BEST** ⚠️

### Gap Between Paper and Our Best Training Results

| Dataset | Paper F1 | Our Best Training | Gap |
|---------|----------|-------------------|-----|
| UCI_HAR | 96.1% | 92.7% | **-3.4%** |
| DSADS | 90.0% | 86.3% | **-3.7%** |
| OPPORTUNITY | 81.6% | 76.9% | **-4.7%** |
| KU-HAR | 97.5% | 92.6% | **-4.9%** |
| PAMAP2 | 79.4% | 76.7% | **-2.7%** |

### Possible Causes
1. **Different hyperparameters** - Paper might use different learning rate, batch size, epochs
2. **Different random seeds** - Paper might report best of multiple runs
3. **Different data preprocessing** - Normalization, augmentation, train/test split
4. **Different dataset versions** - Data might have been updated
5. **Ensemble or post-processing** - Paper might use techniques not in the code

### TSC Results Also Affected

| Dataset | Paper Acc | Collected Acc | Gap |
|---------|-----------|---------------|-----|
| AtrialFibrillation | 46.7% | 20.0% | **-26.7%** |
| MotorImagery | 60.0% | 52.0% | **-8.0%** |
| Heartbeat | 80.0% | 76.6% | **-3.4%** |
| PhonemeSpectra | 30.3% | 30.7% | **+0.4%** ✓ |
| LSST | 60.0% | 52.5% | **-7.5%** |
| PEMS-SF | 91.3% | 79.2% | **-12.1%** |

---

## Issue 3: **WRONG METRIC FOR COMPLEXITY** ❌

### Problem
- **Paper uses**: FLOPs (Floating Point Operations)
- **We collected**: Parameters (model size)

These are different metrics!

### Example (CALANet UCI-HAR)
- Paper: 7.6M FLOPs
- Collected: 0.67M Parameters

### Fix Required
Need to compute FLOPs using a library like `fvcore` or `ptflops`:
```python
from fvcore.nn import FlopCountAnalysis
flops = FlopCountAnalysis(model, input_tensor)
total_flops = flops.total()
```

---

## Issue 4: **MISSING AND MISNAMED MODELS** ⚠️

### Missing Models
- **TapNet**: Mentioned in paper, not in our codebase

### Name Mismatches
- Paper: "T-ResNet" → We have: "resnet"
- Paper: "T-FCN" → We have: "FCN" or "FCN_TSC"
- Paper: "SAGoG" → We have: "SAGOG"

---

## Issue 5: **PAPER USES F1 FOR HAR, ACCURACY FOR TSC** ✓

This part we got correct! The paper uses:
- **HAR**: F1-Score (weighted)
- **TSC**: Accuracy

Our collected metrics have both, but tables need to show the right one.

---

## Issue 6: **REALDISP CATASTROPHIC FAILURE** 🚨

| Model | Paper F1 | Collected F1 |
|-------|----------|--------------|
| CALANet | 98.2% | **1.4%** |

REALDISP results are essentially random (1.4% vs 98.2%). This suggests:
1. Model completely failed to converge
2. Data loading issue
3. Wrong evaluation setup

---

## Recommendations

### CRITICAL (Must Fix)
1. ✅ **Fix checkpoint loading** - Load best model before metrics collection
2. ✅ **Compute FLOPs** - Add FLOPs calculation instead of just parameters
3. ✅ **Investigate REALDISP** - Debug why it's failing catastrophically

### IMPORTANT (Should Fix)
4. ⚠️ **Investigate hyperparameters** - Compare with paper's actual settings
5. ⚠️ **Check data preprocessing** - Verify train/test splits match paper
6. ⚠️ **Add TapNet model** - If it's in the paper, should be in experiments

### OPTIONAL (Nice to Have)
7. 💡 **Multiple seeds** - Run experiments with multiple random seeds
8. 💡 **Document differences** - Note in paper if results differ from reported

---

## Next Steps

1. **Immediate**: Fix the checkpoint loading bug in all model scripts
2. **Short-term**: Add FLOPs calculation to metrics collection
3. **Investigation**: Understand why REALDISP and AtrialFibrillation fail so badly
4. **Long-term**: Re-run all experiments with fixed code

---

## Files Affected

### Need Fixing
- `codes/CALANet_local/run.py` - Load best checkpoint
- `codes/CALANet_local/run_TSC.py` - Load best checkpoint
- `codes/millet/run.py` - Load best checkpoint (if applicable)
- `codes/millet/run_TSC.py` - Load best checkpoint (if applicable)
- `codes/shared_metrics.py` - Add FLOPs calculation

### Tables Need Update
- Use F1-weighted for HAR (not accuracy)
- Use Accuracy for TSC (correct)
- Use FLOPs for complexity (not parameters)
- Correct model names (T-ResNet, T-FCN, SAGoG)
