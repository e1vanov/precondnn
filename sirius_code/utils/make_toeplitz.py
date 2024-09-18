import torch
from make_circulant import make_circulant

def make_toeplitz(tensor, dim=0):
    
    # tensor shape: (..., L)
    length = tensor.shape[-1]
    extended_tensor = torch.zeros((tensor.shape[0], 2 * length - 1))
    extended_tensor[...,:length] = tensor
    extended_tensor[...,length:] = torch.flip(tensor[...,1:], dims=[-1])

    return make_circulant(extended_tensor, dim)[...,:length,:length]
