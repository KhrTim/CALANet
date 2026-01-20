"""
Paper results extracted from the provided LaTeX tables
These are the effectiveness metrics (Accuracy/F1) from the published paper
"""

# HAR results from paper (F1-Score, weighted)
PAPER_HAR_F1 = {
    "CALANet": {
        "UCI_HAR": 96.1,
        "DSADS": 90.0,
        "OPPORTUNITY": 81.6,
        "KU-HAR": 97.5,
        "PAMAP2": 79.4,
        "REALDISP": 98.2
    },
    "RepHAR": {
        "UCI_HAR": 95.1,
        "DSADS": 85.5,
        "OPPORTUNITY": 80.0,
        "KU-HAR": 93.4,
        "PAMAP2": 73.0,
        "REALDISP": 94.7
    },
    "DeepConvLSTM": {
        "UCI_HAR": 91.4,
        "DSADS": 85.5,
        "OPPORTUNITY": 62.0,
        "KU-HAR": 93.5,
        "PAMAP2": 77.3,
        "REALDISP": 91.7
    },
    "Bi-GRU-I": {
        "UCI_HAR": 94.6,
        "DSADS": 85.6,
        "OPPORTUNITY": 77.2,
        "KU-HAR": 94.9,
        "PAMAP2": 71.0,
        "REALDISP": 96.1
    },
    "RevTransformerAttentionHAR": {
        "UCI_HAR": 95.1,
        "DSADS": 87.6,
        "OPPORTUNITY": 78.6,
        "KU-HAR": 97.7,
        "PAMAP2": 79.7,
        "REALDISP": 98.5
    },
    "IF-ConvTransformer2": {
        "UCI_HAR": 95.4,
        "DSADS": 87.5,
        "OPPORTUNITY": 82.2,
        "KU-HAR": 96.4,
        "PAMAP2": 80.1,
        "REALDISP": 97.4
    },
    "millet": {
        "UCI_HAR": 94.7,
        "DSADS": 84.3,
        "OPPORTUNITY": 82.3,
        "KU-HAR": 97.8,
        "PAMAP2": 80.2,
        "REALDISP": 95.1
    },
    "DSN": {
        "UCI_HAR": 95.4,
        "DSADS": 86.4,
        "OPPORTUNITY": 71.8,
        "KU-HAR": 97.1,
        "PAMAP2": 68.8,
        "REALDISP": 97.5
    },
    "SAGOG": {
        "UCI_HAR": 5.6,
        "DSADS": 0.5,
        "OPPORTUNITY": 9.1,
        "KU-HAR": 2.1,
        "PAMAP2": 4.6,
        "REALDISP": 1.4
    },
    "MPTSNet": {
        "UCI_HAR": 73.5,
        "DSADS": 85.0,
        "OPPORTUNITY": 73.4,
        "KU-HAR": 68.2,
        "PAMAP2": 64.6,
        "REALDISP": 89.3
    },
    "MSDL": {
        "UCI_HAR": 81.3,
        "DSADS": 88.2,
        "OPPORTUNITY": 77.5,
        "KU-HAR": 85.4,
        "PAMAP2": 71.8,
        "REALDISP": 90.2
    }
}

# TSC results from paper (Accuracy)
PAPER_TSC_ACCURACY = {
    "CALANet": {
        "AtrialFibrillation": 46.7,
        "MotorImagery": 60.0,
        "Heartbeat": 80.0,
        "PhonemeSpectra": 30.3,
        "LSST": 60.0,
        "PEMS-SF": 91.3
    },
    "resnet": {
        "AtrialFibrillation": 20.0,
        "MotorImagery": 56.0,
        "Heartbeat": 71.6,
        "PhonemeSpectra": 32.5,
        "LSST": 23.2,
        "PEMS-SF": 82.8
    },
    "FCN": {
        "AtrialFibrillation": 20.0,
        "MotorImagery": 59.0,
        "Heartbeat": 72.4,
        "PhonemeSpectra": 28.6,
        "LSST": 33.7,
        "PEMS-SF": 83.2
    },
    "InceptionTime": {
        "AtrialFibrillation": 26.7,
        "MotorImagery": 53.0,
        "Heartbeat": 71.7,
        "PhonemeSpectra": 32.9,
        "LSST": 29.0,
        "PEMS-SF": 88.8
    },
    "millet": {
        "AtrialFibrillation": 16.7,
        "MotorImagery": 58.0,
        "Heartbeat": 75.1,
        "PhonemeSpectra": 31.7,
        "LSST": 60.6,
        "PEMS-SF": 81.5
    },
    "DSN": {
        "AtrialFibrillation": 33.3,
        "MotorImagery": 63.0,
        "Heartbeat": 78.5,
        "PhonemeSpectra": 33.0,
        "LSST": 64.4,
        "PEMS-SF": 82.1
    },
    "SAGOG": {
        "AtrialFibrillation": 46.7,
        "MotorImagery": 52.0,
        "Heartbeat": 72.2,
        "PhonemeSpectra": 2.6,
        "LSST": 31.5,
        "PEMS-SF": 11.6
    },
    "MPTSNet": {
        "AtrialFibrillation": 33.3,
        "MotorImagery": 60.0,
        "Heartbeat": 72.2,
        "PhonemeSpectra": 12.3,
        "LSST": 64.1,
        "PEMS-SF": 91.9
    },
    "MSDL": {
        "AtrialFibrillation": 46.7,
        "MotorImagery": 59.0,
        "Heartbeat": 72.2,
        "PhonemeSpectra": 27.2,
        "LSST": 51.3,
        "PEMS-SF": 67.6
    }
}

# FLOPs from paper (in millions) - for reference
PAPER_HAR_FLOPS = {
    "CALANet": {
        "UCI_HAR": 7.6,
        "DSADS": 8.5,
        "OPPORTUNITY": 19.3,
        "KU-HAR": 29.6,
        "PAMAP2": 74.9,
        "REALDISP": 56.7
    },
    # Add other models if needed for comparison
}

PAPER_TSC_FLOPS = {
    "CALANet": {
        "AtrialFibrillation": 80.5,
        "MotorImagery": 435.0,
        "Heartbeat": 46.6,
        "PhonemeSpectra": 12.6,
        "LSST": 0.98,
        "PEMS-SF": 52.2
    },
    # Add other models if needed for comparison
}
