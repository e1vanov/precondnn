import torch
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

def draw_spectrum(t, c_inv, 
                  h=3, w=3, 
                  path='./img.svg'):

    assert t.shape[0] == h * w
    assert c_inv.shape[0] == h * w

    fig, ax = plt.subplots(h, w, figsize=(3 * h, 3 * w))

    for i in range(h):
        for j in range(w):

            ind = i * w + j

            T = sp.linalg.toeplitz(t[ind])
            C_inv = sp.linalg.circulant(c_inv[ind])

            A = np.eye(t.shape[1]) - T @ C_inv

            lambdas = np.linalg.eigvals(A)

            ax[i][j].set_xlim([-2., 2.])
            ax[i][j].set_ylim([-2., 2.])
            ax[i][j].grid(0.1)
            ax[i][j].scatter(np.real(lambdas), np.imag(lambdas), alpha=0.2)

    plt.tight_layout()
    plt.savefig(path)


def draw_symm_toeplitz_ds_spectrum_distr(ds_path, path='./img.svg'):

    t = torch.load(ds_path).detach().cpu().numpy()

    total_spectrum = []

    for i in range(t.shape[0]):

        T = sp.linalg.toeplitz(t[i])
        curr_spectrum = np.linalg.eigvalsh(T)

        total_spectrum.append(curr_spectrum)

    plt.hist(np.ravel(np.stack(total_spectrum)))
    plt.savefig(path)

