import torch
import torch.nn as nn

from make_circulant import make_circulant
from make_toeplitz import make_toeplitz

class SpecRadLoss(nn.Module):

    # TO DO: implement using torch.fft

    # using rho(A) = lim_n sup \|x\| = 1 \| A^n x \|^1/n

    def __init__(self, n=10, 
                       n_samples=100,
                       alpha_reg=1e-12):

        super().__init__()

        self.n = n
        self.n_samples = n_samples
        self.alpha_reg = alpha_reg

    def forward(self, x, circ):

        # x shape: (N, L)
        length = x.shape[1]

        sphere_points = torch.randn((length, self.n_samples))
        sphere_points /= torch.norm(sphere_points, dim=0)
        sphere_points = sphere_points.repeat(x.shape[0], 1, 1)

        A = make_toeplitz(x, dim=1)
        C = torch.permute(make_circulant(circ, dim=1), (0, 2, 1))

        # sphere_points shape: (L, n_samples)

        y = torch.zeros_like(sphere_points)
        y += sphere_points

        for i in range(self.n):
            y = y - torch.matmul(C, torch.matmul(A, y))

        y = y + self.alpha_reg * sphere_points

        vals = torch.max(torch.norm(y, dim=1) ** (1/self.n), dim=1)[0]
        
        return torch.mean(vals)

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
