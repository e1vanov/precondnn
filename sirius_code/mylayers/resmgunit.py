from mgconv import *
import torch
import torch.nn as nn


class ResMGUnit(nn.Module):

    def __init__(self, depth, 
                       channels):

        super().__init__()

        assert depth == len(channels)
        assert depth >= 2

        self.depth = depth
        self.channels = channels
        
        self.mgconv_1 = MGConv(depth, channels)
        
        self.bns_1 = nn.ModuleList([nn.BatchNorm1d(channels[i]) for i in range(depth)])

        self.acts_1 = nn.ModuleList([nn.ReLU() for i in range(depth)])

        self.mgconv_2 = MGConv(depth, channels)

        self.bns_2 = nn.ModuleList([nn.BatchNorm1d(channels[i]) for i in range(depth)])

    def forward(self, x):

        y = self.mgconv_1(x)

        z = [0 for _ in range(len(y))]
        for i in range(self.depth):
            z[i] = self.bns_1[i](y[i])

        w = [0 for _ in range(len(z))]
        for i in range(self.depth):
            w[i] = self.acts_1[i](z[i])

        w = self.mgconv_2(w)

        u = [0 for _ in range(len(w))]
        for i in range(self.depth):
            u[i] = self.bns_2[i](w[i])

        for i in range(self.depth):
            u[i] += x[i]

        return u
