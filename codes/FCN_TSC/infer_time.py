import warnings
warnings.filterwarnings('ignore')

import torch
from torch.autograd import Variable

from model import FCN_TSC
import numpy as np
from utils import data_info

#dataset = "UCI_HAR"
#dataset = "WISDM"
#dataset = "OPPORTUNITY"
dataset = "PAMAP2"
#dataset = "UniMiB-SHAR"

input_nc, segment_size, class_num = data_info(dataset)

device = torch.device("cpu") # cuda

model = FCN_TSC(input_nc, class_num, segment_size)
model.eval()
model.to(device)

x = Variable(torch.rand(1,input_nc, segment_size)).to(device)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions= 1000
timings=np.zeros((repetitions,1))

#DEVICE-WARM-UP
for _ in range(10):
    _ = model(x)

with torch.no_grad():
    for i in range(repetitions):
        starter.record()
        _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        timings[i] = starter.elapsed_time(ender)
    
print("Processing time : %.2f s" % (np.sum(timings/1000)))
print("Min : %.2f ms" % (np.min(timings)))
print("Max : %.2f ms" % (np.max(timings)))