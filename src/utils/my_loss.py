import torch
import torch.nn as nn

from math import pi

from matrix_builder import (make_circulant,
                            make_symm_toeplitz,
                            make_tc_inv,
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

        raise NotImplementedError('Not ready for arbitrary optimization!')

        # TO DO: change to log computations

        matrices = make_eye_sub_tc_inv(t, c_inv)
        
        d = matrices.shape[-1]

        # https://discuss.pytorch.org/t/how-to-calculate-matrix-trace-in-3d-tensor/132435/2
        traces = matrices.diagonal(offset=0, 
                                   dim1=-2, 
                                   dim2=-1).sum(dim=-1)

        dets = torch.linalg.det(matrices)
        
        return torch.mean((traces / d) ** d / dets)

class GershgorinPolyRegressionLoss(nn.Module):

    def __init__(self, 
                 distr=lambda: 1 + torch.poisson(torch.tensor(1.)), 
                 rad_num=20,
                 arg_num=20,
                 alpha=1e-2,
                 beta=1e-2,
                 lambda_reg=1e-6,
                 strategy='exact',
                 num_steps=5,
                 approx_lr=1e-3):

        super().__init__()
        self.distr = distr

        self.rad_num = rad_num
        self.arg_num = arg_num

        self.alpha = alpha
        self.beta = beta 

        self.lambda_reg = lambda_reg
        self.strategy = strategy
        self.num_steps = num_steps
        self.approx_lr = approx_lr

    def forward(self, t, c_inv):

        k = int(self.distr())

        n, d = t.shape
        A = make_tc_inv(t, c_inv) # (n, d, d)

        center = torch.diagonal(A, dim1=-2, dim2=-1) # (n, d)
        radius = torch.sum(torch.abs(A - torch.diag_embed(center)), dim=-1) # (n, d)
        
        # ===

        unit_rad = torch.linspace(0., 1., 
                                  self.rad_num + 1,
                                  requires_grad=True)[1:]

        unit_arg = torch.linspace(0., 2 * pi,
                                  self.arg_num + 1,
                                  requires_grad=True)[:-1]

        unit_point = torch.outer(unit_rad, torch.exp(1.j * unit_arg)).view(-1) # (rad_num * arg_num)

        # ===

        point = (center[..., None] + \
                 radius[..., None] * unit_point[None, None, ...]).view(n, -1) # (n, d * rad_num * arg_num)

        X = torch.linalg.vander(point, N=k+1) # (n, d * rad_num * arg_num, k + 1)
        X, y = X[..., 1:], X[..., 0] # X: (n, d * rad_num * arg_num, k), y: (n, d * rad_num * arg_num)

        if self.strategy == 'exact':

            X_conj = torch.transpose(torch.conj_physical(X), -2, -1) # (n, k, d * rad_num * arg_num)
            
            # r = X (X* X)_inv X* y - y
            u = torch.matmul(X_conj, y[..., None])
            v = torch.linalg.solve(torch.matmul(X_conj, X) + self.lambda_reg * torch.eye(k).repeat(n, 1, 1), u)
            r = torch.squeeze(torch.matmul(X, v)) - y

        elif self.strategy == 'torch_lstsq':

            coeffs = torch.linalg.lstsq(X, y, driver='gelsy').solution
            r = torch.squeeze(torch.matmul(X, coeffs[..., None])) - y

        elif self.strategy == 'approx':

            X_mean = torch.mean(X, dim=-2) # (n, k)
            X_std = torch.std(X, dim=-2) # (n, k)
            X = (X - X_mean[:, None, :]) / X_std[:, None, :]

            X_conj = torch.transpose(torch.conj_physical(X), -2, -1) # (n, k, d * rad_num * arg_num)

            a = torch.ones(k).type(torch.complex64)

            for _ in range(self.num_steps):

                u = torch.matmul(X_conj, y[..., None])
                v = self.lambda_reg * a
                w = torch.matmul(X_conj, torch.matmul(X, a[..., None]))

                a = a - self.approx_lr * 2 *  (torch.squeeze(w) + v - torch.squeeze(u))

            r = torch.squeeze(torch.matmul(X, a[..., None])) - y

        # r: (n, d * rad_num * arg_num)
        loss_part1 = torch.mean(torch.sum(torch.abs(r) ** 2, axis=1) / r.shape[1])
        loss_part2 = torch.mean((torch.abs(center) - 1) ** 2)

        return loss_part1 + self.alpha * loss_part2 
