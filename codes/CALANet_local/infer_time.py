import warnings
warnings.filterwarnings('ignore')

import torch
from torch.autograd import Variable

from models import HTAggNet
import numpy as np
from utils import data_info

#dataset = "UCI_HAR"
#dataset = "WISDM"
#dataset = "OPPORTUNITY"
#dataset = "PAMAP2"
#dataset = "UniMiB-SHAR"
#dataset = "KU-HAR"
dataset = "REALDISP"

input_nc, segment_size, class_num = data_info(dataset)
L = 8

device = torch.device("cpu") # cuda

model = HTAggNet(input_nc, class_num, segment_size, L)
model.eval()
model.to(device)

x = Variable(torch.rand(1,input_nc, segment_size)).to(device)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions= 10000
timings=np.zeros((repetitions,1))

#DEVICE-WARM-UP
for _ in range(100):
    _ = model(x)

with torch.no_grad():
    for i in range(repetitions):
        starter.record()
        _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        timings[i] = starter.elapsed_time(ender)
    
#mean_syn = np.sum(timings) / repetitions
#std_syn = np.std(timings)
print("Processing time : %.2f ms" % (np.sum(timings)/repetitions))
print("Min : %.2f ms" % (np.min(timings)))
print("Max : %.2f ms" % (np.max(timings)))