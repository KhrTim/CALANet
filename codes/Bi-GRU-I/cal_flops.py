from models import Bi_GRU_I
import torch
from thop import profile, clever_format
from utils import data_info

#dataset = "UCI_HAR"
#dataset = "WISDM"
dataset = "OPPORTUNITY"
#dataset = "PAMAP2"
#dataset = "UniMiB-SHAR"
#dataset = "DSADS"
#dataset = "KU-HAR"
#dataset = "REALDISP"

input_nc, segment_size, class_num = data_info(dataset)

net = Bi_GRU_I(input_nc, class_num)
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