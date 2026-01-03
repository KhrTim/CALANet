from models import TapNet
import torch
from thop import profile, clever_format
import math

input_nc = 963
segment_size = 144
class_num =7

net = TapNet(nfeat=input_nc,
                   len_ts=segment_size,
                   layers=[500,300],
                   nclass=class_num,
                   dropout=0.5,
                   use_lstm=True,
                   use_cnn=True,
                   filters=[256,256,128],
                   dilation=1,
                   kernels=[8,5,3],
                   use_ss=False,
                   use_metric=False,
                   use_rp=True,
                   rp_params=[3, math.floor(input_nc * 2 /3)]
                   )
ts = torch.autograd.Variable(torch.rand(1, input_nc, segment_size))
labels = torch.LongTensor(0)
idx_train = torch.LongTensor(0)

input_tensor = (ts, labels, idx_train)

macs, params = profile(net, inputs=(input_tensor,))
flops, params = clever_format([macs*2, params], "%.3f")
print("FLOPs: ", flops , "; #params: ", params)

# with torch.cuda.device(0):
#     net = HAR_ConvLSTM(input_nc, class_num)
#     macs, params = get_model_complexity_info(net, (input_nc,segment_size), as_strings=False,
#                                            print_per_layer_stat=True, verbose=True)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs/(10e+5)))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params/(10e+5)))