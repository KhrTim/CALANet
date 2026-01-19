import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def channel_shuffle(x, groups):
    batchsize, num_channels, T = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, 
        channels_per_group, T)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, T)
    return x

def pool_per_T(T,L):
    #if T>=512:
    #    pool_num = 4
    if T>=256:
        pool_num = 3
    elif T>=128:
        pool_num = 2
    elif T>=64:
        pool_num = 1
    else:
        return []

    split = [int(np.floor(L / (pool_num+1))) for _ in range(pool_num+1)]
    residual = L - sum(split)
    for i in range(residual):
        split[i] += 1
    split.pop()

    pool_idx = []
    pidx = -1
    for i in split:
        pidx += i
        pool_idx.append(pidx)
    return pool_idx

class InplaceShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_groups):
        ctx.groups_ = n_groups
        n, c, t = input.size()
        slide = c // n_groups
        left_idx = torch.tensor([i*slide for i in range(n_groups)])
        right_idx = torch.tensor([i*slide+1 for i in range(n_groups)])

        buffer = input.data.new(n, n_groups, t).zero_()
        buffer[:, :, :-1] = input.data[:, left_idx, 1:] 
        input.data[:, left_idx] = buffer
        buffer.zero_()
        buffer[:, :, 1:] = input.data[:, right_idx, :-1]
        input.data[:, right_idx] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        n_groups = ctx.groups_
        n, c, t = grad_output.size()
        slide = c // n_groups
        left_idx = torch.tensor([i*slide for i in range(n_groups)])
        right_idx = torch.tensor([i*slide+1 for i in range(n_groups)])

        buffer = grad_output.data.new(n, left_idx,t).zero_()
        buffer[:, :, 1:] = grad_output.data[:, left_idx, :-1] # reverse
        grad_output.data[:, left_idx] = buffer
        buffer.zero_()
        buffer[:, :, :-1] = grad_output.data[:, right_idx, 1:]
        grad_output.data[:, right_idx] = buffer
        return grad_output, None

class GTSConv(nn.Module):
    def __init__(self, i_nc, n_groups):
        super(GTSConv, self).__init__()
        self.groups = n_groups
        self.conv = nn.Conv1d(i_nc, i_nc, kernel_size=1, padding=0, bias=False, groups=n_groups)
        self.bn = nn.BatchNorm1d(i_nc)
    def forward(self, x):
        out = InplaceShift.apply(x, self.groups)
        out = self.conv(x)
        out = self.bn(out)
        return out
    
class GTSConvUnit(nn.Module):
    '''
    Grouped Temporal Shift (GTS) module
    '''
    def __init__(self, i_nc, n_fs, n_groups, first_grouped_conv=True):
        super(GTSConvUnit, self).__init__()

        self.groups = n_groups
        self.grouped_conv = n_groups if first_grouped_conv else 1

        self.perm = nn.Sequential(
            nn.Conv1d(i_nc, n_fs, kernel_size=1, groups=self.grouped_conv, stride=1, bias=False),
            nn.BatchNorm1d(n_fs),
        )

        self.GTSConv = GTSConv(n_fs, n_groups)
        
    def forward(self, x):
        out = F.relu(self.perm(x))
        out = self.GTSConv(out)
        out = channel_shuffle(out, self.groups)
        return out


class TAggBlock(nn.Module):
    def __init__(self, i_nc, o_nc, L, T, pool):
        super(TAggBlock, self).__init__()
        self.L = L
        self.pool = pool

        # local temperol convolution
        #self.gconv = nn.Conv1d(i_nc, o_nc, kernel_size=5, padding='same', bias=False, groups=L)
        # n_groups=4 matches original paper architecture (not o_nc//4 which would be 16)
        self.gconv = GTSConvUnit(i_nc, o_nc, 4)

        # temporal glance convolution
        self.tgconv = nn.Conv1d(o_nc, o_nc//L, kernel_size=T)
        
    def forward(self, x):
        _,_,t = x.size()
        if self.pool:
            x = F.adaptive_max_pool1d(x,t//2)
        x = self.gconv(x)
        l_feat = F.relu_(x)
        #l_feat = channel_shuffle(x, self.L)
        g_feat = self.tgconv(l_feat)
        return l_feat, g_feat

class HTAggNet(nn.Module):
    def __init__(self, nc_input, n_classes, segment_size, L):

        super(HTAggNet, self).__init__()
        T = segment_size
        nc_o = 64
        if nc_input > 64:
            nc_o *= 2
            pre_mul = 1
        else:
            pre_mul = 0
        self.L = L

        self.stem = nn.Conv1d(nc_input, nc_o, 5, padding='same')   
        #self.bn = nn.BatchNorm1d(nc_o)     
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        pool_idx = pool_per_T(T,L)

        T = math.ceil(T / 2)
        nc_i = nc_o

        out_channel_num = []

        self.layers = nn.ModuleList()
        for i in range(self.L):
            if i in pool_idx:
                pool = True
                T = T // 2
                if pre_mul > 0:
                    pre_mul -=1
                else:
                    nc_o *= 2
            else:
                pool = False
            self.layers.append(TAggBlock(nc_i, nc_o, self.L, T, pool))
            nc_i = nc_o
            out_channel_num.append(nc_o//L)
        
        self.fc = nn.Linear(sum(out_channel_num), n_classes)
    
    def forward(self, x):
        #x = self.bn(self.stem(x))
        x = self.stem(x)
        x = self.maxpool(x)

        #fm = []
        out = []
        for block in self.layers:
            x, g_feat = block(x)
            #fm.append(x)
            out.append(g_feat)
        
        out = torch.cat(out, dim=1)

        out = out.view(out.size(0), -1)

        logits = self.fc(out)

        return F.log_softmax(logits, dim=1)#, fm