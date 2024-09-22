import torch
from torch.fft import fft, ifft

def fast_circ_matvec(circ, v):

    # circ shape: (N, D)
    # v shape: (N, D, n_samples)
    # out shape: (N, D, n_samples)

    # usual solution: circ_matrices (N, D, D) -> matvec -> (N, D, n_samples)

    circ_fd = ifft(circ, norm='forward') 
    v_fd = ifft(v, dim=-2)

    return torch.real(fft(torch.einsum('ij,ijk->ijk', circ_fd, v_fd), dim=1))
