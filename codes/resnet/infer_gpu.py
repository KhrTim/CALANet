import warnings
warnings.filterwarnings('ignore')

import torch
from torch.autograd import Variable

from model import ResNet_TSC
import time
import torch.backends.cudnn as cudnn
import sys
from utils import data_info

dataset = "UCI_HAR"
#dataset = "WISDM"
#dataset = "OPPORTUNITY"
#dataset = "PAMAP2"
#dataset = "UniMiB-SHAR"

input_nc, segment_size, class_num = data_info(dataset)

if not torch.cuda.is_available():
    print('no gpu device available')
    sys.exit(1)

torch.cuda.set_device(0)
cudnn.benchmark = True # This automatically benchmarks each possible algorithm on your GPU and chooses the fastest one.

model = ResNet_TSC(input_nc, class_num).cuda()
#device = torch.device(0)
#model.to(device)
model.eval()


repetitions= 1000
total_time = 0

with torch.no_grad():
    for i in range(repetitions):
        x = torch.randn(1,input_nc, segment_size, dtype=torch.float).cuda()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        total_time += starter.elapsed_time(ender)/1000
    
print("Processing time : %.2f ms" % (total_time/repetitions*1000))