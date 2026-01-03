Previously we had a revision where we had to add new methods into paper and we included SAGoG, MPTSNET and MSDL.

We received a new revision where a reviewer asked to add evaluation metrics into experiments:
"1.The evaluation metrics are too limited, making it difficult to comprehensively assess the effectiveness and efficiency of the proposed model. It is recommended that the authors include additional metrics.
Effectiveness metrics: Accuracy, Precision, Recall, Confusion Matrix, and statistical significance tests (e.g., t-test or Wilcoxon).
Efficiency metrics: training time, inference time, throughput (samples/s), peak GPU memory usage, and number of parameters."


So I added into codes all previously used models for the counterpart experiments.
for HAR:
RepHAR, DeepConvLSTM, Bi-GRU-I, RevAttNet, IF-ConvTransf, millet, DSN,SAGoG, MPTSNET, MSDL, Calanet
for TSC:
T-ResNet (resnet), T-FCN, InceptionTime, TapNet, millet, DSN, SAGoG, MPTSNET, MSDL, Calanet.

Can you review existing codebase and extend experiments to include more metrics?
