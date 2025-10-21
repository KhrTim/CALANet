**CALANet_local** contains files of my proposed method implementation. Use it as a reference for working with datasets.

# My current goal:
I need to add several more algorithms for comparison.

# Comparison datasets:
I need to conduct experiemtns on 2 groups of datasets:
- ## HAR:
UCI-HAR; DSADS; OPPORTUNITY;KU-HAR; PAMAP2; REALDISP
### Metrics: 
F1, Flops
- ## TSC:
AtrialFibrillation; MotorImagery; Heartbeat;PhonemeSpectra; LSST; PEMS-SF
### Metrics: 
Accuracy, Flops

# Additional models and implementation information

All models have been implemented with claude code previously but some of them might need some refinement because results are too far from presented in papers or Flops are inconsistent.

All papers are downloaded and placed in corresponding model directories for your review.

- ## SAGOG
### Paper title: SAGoG: Similarity-Aware Graph of Graphs Neural Networks for Multivariate Time Series Classification

### Additional info:
I requested analysis of the model before and I copied it into a file "implementation_discrepancies.md".
Also I noticed that in paper authors follow an approach "Dynamic Time Warping (DTW)" which is used in another paper, which has a github implementation "https://github.com/daochenzha/SimTSC". You may take a look at it as well if you think its helpful.
Lastly, Sagog paper has 3 datasets same with ours and results are as follows: Heartbeat - 0.761; MotorImagery- 0.672; PhonemeSpectra- 0.341. Metric- accuracy. You can use these values as a reference to know that your implementation is correct.

### Action to take:
Review the implementation and make sure that it's as close to the description as you can get.


- ## GTWIDL
### Paper title: Generalized time warping invariant dictionary learning for time series classification and clustering

### Additional info:
Paper has an algorithm description in "Algorithm 1" section

### Action to take:
Review the implementation and make sure that it's as close to the description as you can get.


- ## MPTSNet
### Paper title: MPTSNet: Integrating multiscale periodic local patterns and global dependencies for multivariate time series classification

### Additional info:
Has an official code released: https://github.com/MUYang99/MPTSNet.
Has 2 datasets same with ours and results are as follows: Heartbeat - 0.756; PhonemeSpectra- 0.144.You can use these values as a reference to know that your implementation is correct.
### Action to take:
Review the implementation and make sure that it's as close to the description as you can get.


- ## MSDL

### Paper title: Multiscale temporal dynamic learning for time series classification

### Action to take:
Review the implementation and make sure that it's as close to the description as you can get.


# WORKFLOW DETAILS:
1. I need to create pdf reports when all resutls are collected
2. For each logical step **MAKE SURE TO COMMIT YOUR CHANGES** because we can keep track of the process.
3. After you prepare experiment running skripts I would like to parallelize the execution to speed up the process as I don't have much time left. I will copy files to multi-gpu workstation with 3 or 4 RTX 3090, so make sure to add some parallel training- split by models or datsets (choose easiest path).





