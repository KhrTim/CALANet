from model import ResNet_TSC
import torch
from thop import profile, clever_format
from utils import data_info

#dataset = "UCI_HAR"
#dataset = "WISDM"
#dataset = "OPPORTUNITY"
#dataset = "PAMAP2"
#dataset = "UniMiB-SHAR"
#dataset = "DSADS"
dataset = "KU-HAR"
#dataset = "REALDISP"

input_nc, segment_size, class_num = data_info(dataset)

net = ResNet_TSC(input_nc, class_num)
input_tensor = torch.autograd.Variable(torch.rand(1, input_nc, segment_size))

macs, params = profile(net, inputs=(input_tensor,))
flops, params = clever_format([macs*2, params], "%.3f")
print("FLOPs: ", flops , "; #params: ", params)