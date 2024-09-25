import torch
import torch.nn as nn

from matrix_builder import (make_circulant,
                            make_symm_toeplitz,
                            make_eye_sub_tc_inv)

from fast_circ_ops import fast_circ_matvec

class SpecRadLoss(nn.Module):

    # using rho(A) = lim_n sup \|x\| = 1 \| A^n x \|^1/n

    def __init__(self, n=10, 
                       n_samples=100,
                       alpha_reg=1e-12):

        super().__init__()

        self.n = n
        self.n_samples = n_samples
        self.alpha_reg = alpha_reg

    def forward(self, t, c_inv):

        # x shape: (N, D)
        length = t.shape[1]

        sphere_points = torch.randn((length, self.n_samples))
        sphere_points /= torch.norm(sphere_points, dim=0)
        sphere_points = sphere_points.repeat(t.shape[0], 1, 1)

        T = make_symm_toeplitz(t, dim=1)

        # sphere_points shape: (N, D, n_samples)

        y = torch.zeros_like(sphere_points)
        y += sphere_points

        for i in range(self.n):
            y = y - torch.matmul(T, fast_circ_matvec(c_inv, y))

        y = y + self.alpha_reg * sphere_points

        vals = torch.max(torch.norm(y, dim=1) ** (1/self.n), dim=1)[0]
        
        return torch.mean(vals)

class MatrixNormLoss(nn.Module):

    def __init__(self, p):

        super().__init__()
        self.p = p

    def forward(self, t, c_inv):

        return torch.mean(torch.linalg.matrix_norm(make_eye_sub_tc_inv(t, c_inv), ord=self.p))

class KCondNumberLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, t, c_inv):

        matrices = make_eye_sub_tc_inv(t, c_inv)
        
        d = matrices.shape[-1]

        # https://discuss.pytorch.org/t/how-to-calculate-matrix-trace-in-3d-tensor/132435/2
        traces = matrices.diagonal(offset=0, 
                                   dim1=-2, 
                                   dim2=-1).sum(dim=-1)

        dets = torch.linalg.det(matrices)
        
        return torch.mean((traces / d) ** d / dets)
