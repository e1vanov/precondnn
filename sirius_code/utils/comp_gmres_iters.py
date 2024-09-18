import torch
import numpy as np
import scipy as sp

class gmres_counter():
    def __init__(self):
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1

@torch.no_grad()
def comp_gmres_iters(model,
                     matrix_generator):

    matrices = np.vstack([val for val in matrix_generator])
    circs = model(torch.tensor(matrices)[:,None,:]).cpu().numpy()

    for mat, circ in zip(matrices, circs):

        A = sp.linalg.toeplitz(mat)
        C = sp.linalg.circulant(circ)

        x = 3 * np.random.randn(matrices.shape[-1])
        b = A @ x

        counter = gmres_counter()
        x_pred_with, info_with = sp.sparse.linalg.lgmres(A, b, M=C,
                                                         callback=counter)
        print("With: ", counter.niter)

        counter = gmres_counter()
        x_pred_with, info_with = sp.sparse.linalg.lgmres(A, b,
                                                         callback=counter)
        print("Without: ", counter.niter)
