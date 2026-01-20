import torch
from thop import profile, clever_format
from utils import data_info
from models import Inception, InceptionBlock
import torch.nn as nn


input_nc = 963
segment_size = 144
class_num = 7

class Flatten(nn.Module):
	def __init__(self, out_features):
		super(Flatten, self).__init__()
		self.output_dim = out_features

	def forward(self, x):
		return x.view(-1, self.output_dim)
	
model = nn.Sequential(
                    InceptionBlock(
                        in_channels=input_nc, 
                        n_filters=32, 
                        kernel_sizes=[5, 11, 23],
                        bottleneck_channels=32,
                        use_residual=True,
                        activation=nn.ReLU()
                    ),
                    InceptionBlock(
                        in_channels=32*4, 
                        n_filters=32, 
                        kernel_sizes=[5, 11, 23],
                        bottleneck_channels=32,
                        use_residual=True,
                        activation=nn.ReLU()
                    ),
                    nn.AdaptiveAvgPool1d(output_size=1),
                    Flatten(out_features=32*4*1),
                    nn.Linear(in_features=4*32*1, out_features=class_num)
        )
input_tensor = torch.autograd.Variable(torch.rand(1, input_nc, segment_size))

macs, params = profile(model, inputs=(input_tensor,))
flops, params = clever_format([macs*2, params], "%.3f")
print("FLOPs: ", flops , "; #params: ", params)

# with torch.cuda.device(0):
#     net = HAR_ConvLSTM(input_nc, class_num)
#     macs, params = get_model_complexity_info(net, (input_nc,segment_size), as_strings=False,
#                                            print_per_layer_stat=True, verbose=True)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs/(10e+5)))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params/(10e+5)))