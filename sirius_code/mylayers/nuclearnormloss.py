import torch
import torch.nn as nn

import sys
sys.path.append('../utils')

from make_circulant import make_circulant
from make_toeplitz import make_toeplitz

class NuclearNormLoss(nn.Module):

    # using rho(A) = lim_n sup \|x\| = 1 \| A^n x \|^1/n

    def __init__(self):

        super().__init__()

    def forward(self, x, circ):

        # x shape: (N, L)
        length = x.shape[1]

        A = make_toeplitz(x, dim=1)
        C = torch.permute(make_circulant(circ, dim=1), (0, 2, 1))

        vals = torch.norm(torch.eye(length).repeat(x.shape[0], 1, 1) - torch.matmul(A, C),
                          p='nuc',
                          dim=(1,2))
        
        return torch.mean(vals)
