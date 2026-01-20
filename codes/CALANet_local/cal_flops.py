from models import HTAggNet
import torch
from thop import profile, clever_format
from utils import data_info


input_nc = 64
segment_size = 3000
class_num = 2

L = 8

net = HTAggNet(input_nc, class_num, segment_size, L)
input_tensor = torch.autograd.Variable(torch.rand(1, input_nc, segment_size))

macs, params = profile(net, inputs=(input_tensor,))
flops, params = clever_format([macs*2, params], "%.3f")
print("FLOPs: ", flops , "; #params: ", params)

# with torch.cuda.device(0):
#     net = HAR_ConvLSTM(input_nc, class_num)
#     macs, params = get_model_complexity_info(net, (input_nc,segment_size), as_strings=False,
#                                            print_per_layer_stat=True, verbose=True)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs/(10e+5)))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params/(10e+5)))