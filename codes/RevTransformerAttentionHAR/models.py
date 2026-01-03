import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import math

class GaussianDropout(nn.Module):

    def __init__(self, p: float):
        """
        Multiplicative Gaussian Noise dropout with N(1, p/(1-p))
        It is NOT (1-p)/p like in the paper, because here the
        noise actually increases with p. (It can create the same
        noise as the paper, but with reversed p values)

        Source:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

        :param p: float - determines the the standard deviation of the
        gaussian noise, where sigma = p/(1-p).
        """
        super().__init__()
        assert 0 <= p < 1
        self.t_mean = torch.ones((0,))
        self.shape = ()
        self.p = p
        self.t_std = self.compute_std()

    def compute_std(self):
        return self.p / (1 - self.p)

    def forward(self, t_hidden):
        if self.training and self.p > 0.:
            if self.t_mean.shape != t_hidden.shape:
                self.t_mean = torch.ones_like(input=t_hidden
                                              , dtype=t_hidden.dtype
                                              , device=t_hidden.device)
            elif self.t_mean.device != t_hidden.device:
                self.t_mean = self.t_mean.to(device=t_hidden.device, dtype=t_hidden.dtype)

            t_gaussian_noise = torch.normal(self.t_mean, self.t_std)
            t_hidden = t_hidden.mul(t_gaussian_noise)
        return t_hidden

class SelfAttention(nn.Module):
    def __init__(self, k, heads = 8, drop_rate = 0):
        super(SelfAttention, self).__init__()
        self.k, self.heads = k, heads
        # map k-dimentional input to k*heads dimentions
        self.tokeys    = nn.Linear(k, k * heads, bias = False)
        self.toqueries = nn.Linear(k, k * heads, bias = False)
        self.tovalues  = nn.Linear(k, k * heads, bias = False)
        # set dropout rate
        self.dropout_attention = nn.Dropout(drop_rate)
        # squeeze to k dimentions through linear transformation
        self.unifyheads = nn.Linear(heads * k, k)
        
    def forward(self, x):
        
        b, t, k = x.size()
        h = self.heads
        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x).view(b, t, h, k)
        values  = self.tovalues(x).view(b, t, h, k)
        # squeeze head into batch dimension
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys    = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values  = values.transpose(1, 2).contiguous().view(b * h, t, k)
        # normalize the dot products
        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))
        # matrix multiplication
        dot  = torch.bmm(queries, keys.transpose(1,2))
        # softmax normalization
        dot = F.softmax(dot, dim=2)
        dot = self.dropout_attention(dot)
        out = torch.bmm(dot, values).view(b, h, t, k)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h*k)
        
        return self.unifyheads(out) # (b, t, k)
    
# class TransformerBlock(nn.Module):
#     def __init__(self, k, heads, drop_rate=0.0):
#         super(TransformerBlock, self).__init__()

#         self.attention = SelfAttention(k, heads = heads, drop_rate = drop_rate)
#         self.norm1 = nn.LayerNorm(k)

#         self.mlp = nn.Sequential(
#             nn.Linear(k, 4*k),
#             nn.ReLU(),
#             nn.Linear(4*k, k)
#         )
#         self.norm2 = nn.LayerNorm(k)
#         self.dropout_forward = nn.Dropout(drop_rate)

#     def forward(self, x):
        
#         # perform self-attention
#         attended = self.attention(x)
#         # perform layer norm
#         x = self.norm1(attended + x)
#         # feedforward and layer norm
#         feedforward = self.mlp(x)
        
#         return self.dropout_forward(self.norm2(feedforward + x))
    
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class RevTransformerAttentionHAR(nn.Module):
    def __init__(self, nc_input, n_classes, T):
        
        super(RevTransformerAttentionHAR, self).__init__()

        self.layer = nn.Sequential(
            TimeDistributed(nn.Flatten(), batch_first=True),

        )


        self.conv1 = nn.Sequential(
            nn.Conv1d(nc_input, 16, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=5, padding='same', bias=False),
            nn.BatchNorm1d(32),
            GaussianDropout(0.1)
        )
        
        ##block 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm1d(48),
            nn.Conv1d(48, 64, kernel_size=9, padding='same', bias=False),
            nn.BatchNorm1d(64),
            GaussianDropout(0.1)
        )

        self.rnn1 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        
        self.convT1 = nn.Sequential(
            nn.ConvTranspose1d(32,32,15,padding=7),
            nn.ReLU(),
        )

        self.mha1 = SelfAttention(32,7)

        ##block 3
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=11, padding='same', bias=False),
            nn.BatchNorm1d(96),
            nn.Conv1d(96, 128, kernel_size=13, padding='same', bias=False),
            nn.BatchNorm1d(128),
            GaussianDropout(0.1)
        )

        self.rnn2 = nn.LSTM(input_size=128, hidden_size=32, num_layers=1, batch_first=True)
        
        self.convT2 = nn.Sequential(
            nn.ConvTranspose1d(32,64,23,padding=11),
            nn.ReLU(),
        )

        self.mha2 = SelfAttention(64, 10)
        
        self.linear = nn.Linear(224, n_classes)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.layer(x)
        x = torch.transpose(x, 1, 2)
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        h2,_ = self.rnn1(torch.transpose(x2, 1, 2))
        h2 = torch.transpose(h2, 1, 2)
        h2 = self.convT1(h2)
        attn2 = self.mha1(torch.transpose(x1+h2,1,2))
        attn2 = torch.transpose(attn2,1,2)

        #print(x2.size())
        x3 = self.conv3(x2)
        h3,_ = self.rnn2(torch.transpose(x3, 1, 2))
        h3 = torch.transpose(h3, 1, 2)
        h3 = self.convT2(h3)
        attn3 = self.mha2(torch.transpose(x2 + h3,1,2))
        attn3 = torch.transpose(attn3,1,2)


        max1 = F.adaptive_max_pool1d(attn2 + x1, 1)
        max2 = F.adaptive_max_pool1d(attn3 + x2, 1)
        max3 = F.adaptive_max_pool1d(x3, 1)


        feature = torch.concat([max1,max2,max3], dim=1).squeeze(2)
        

        output = self.linear(feature)
        
        
        return output
  

