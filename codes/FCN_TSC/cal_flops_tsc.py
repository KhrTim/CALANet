from model import FCN_TSC
import torch
from thop import profile, clever_format
from utils import data_info

input_nc =6
segment_size = 36
class_num = 14

net = FCN_TSC(input_nc, class_num)
input_tensor = torch.autograd.Variable(torch.rand(1, input_nc, segment_size))

macs, params = profile(net, inputs=(input_tensor,))
flops, params = clever_format([macs*2, params], "%.3f")
print("FLOPs: ", flops , "; #params: ", params)