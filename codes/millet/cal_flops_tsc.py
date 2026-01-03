from model.millet_model import MILLETModel
from model import backbone, pooling
import torch
from thop import profile, clever_format
from utils import data_info
import torch.nn as nn

input_nc = 963
segment_size = 144
class_num = 7

# Pooling config
d_in = 128  # Output size of feature extractor
n_clz = class_num
dropout = 0.1
apply_positional_encoding = True

device = torch.device("cpu")

# Example network
class ExampleNet(nn.Module):
    def __init__(self, feature_extractor, pool):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pool = pool

    def forward(self, bags, pos=None):
        timestep_embeddings = self.feature_extractor(bags)
        return self.pool(timestep_embeddings, pos=pos)

# Create network using InceptionTime feature extractor and Conjunctive Pooling
net = ExampleNet(
    backbone.InceptionTimeFeatureExtractor(input_nc),
    pooling.MILConjunctivePooling(
        d_in,
        n_clz,
        dropout=dropout,
        apply_positional_encoding=apply_positional_encoding,
    ),
)



#net = MILLETModel("ExampleNet", device, n_clz, net)


input_tensor = torch.autograd.Variable(torch.rand(1, input_nc, segment_size))

macs, params = profile(net, inputs=(input_tensor,))
flops, params = clever_format([macs*2, params], "%.3f")
print("FLOPs: ", flops , "; #params: ", params)
