from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from networks.SoftMedian import softMedian, softMin, softMax

################################################### MY ####################
class ResBlock(nn.Module):
    def __init__(self, din, n_block=1):
        super().__init__()
        self.n_block = n_block
        self.res_convs = nn.ModuleList()
        self.prelus = nn.ModuleList()
        for i in range(self.n_block):
            self.res_convs.append(
                nn.Sequential(
                    nn.Conv2d(din, din, 3, 1, 1),
                    nn.PReLU(),
                    nn.Conv2d(din, din, 3, 1, 1),
                    nn.PReLU()
                )
            )
            self.prelus.append(nn.PReLU())

    def forward(self, x):
        for i in range(self.n_block):
            resx = x
            x = self.prelus[i](self.res_convs[i](x)) + resx
        return x

class Model(nn.Module):
    def __init__(self, dim=64, inch=21, outch=3, ifSoftMedian=False):
        super().__init__()
        self.ifSoftMedian = ifSoftMedian
        self.input = nn.Conv2d(inch, dim, 3, 1, 1)
        self.res1 = ResBlock(dim, 20)
        
        if not ifSoftMedian:
            self.out = nn.Conv2d(dim, outch, 3, 1, 1)
        else:
            self.out = nn.Conv2d(dim, inch, 3, 1, 1)
            self.reduce = softMedian

    def forward(self, x):
        bn, ch, sq, h, w = x.shape
        input_ = x.view(bn, -1, h, w)
        input_ = self.input(input_)
        out1 = self.res1(input_)
        if self.ifSoftMedian:
            out1 = self.out(out1)
            out1 = out1.view(bn, ch, sq, h, w)
            out = self.reduce(out1, dim=2)
        else:
            out = self.out(out1)
        return out + x[:,:,3], []
