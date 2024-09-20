import torch

def make_circulant(tensor, 
                   dim=0):

    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), 
                                                       dim=dim, 
                                                       start=0, 
                                                       length=S-1)], 
                    dim=dim)

    return tmp.unfold(dim, S, 1).flip((-1,))

def make_toeplitz(tensor, dim=0):
    
    # tensor shape: (..., L)
    length = tensor.shape[-1]
    extended_tensor = torch.zeros((tensor.shape[0], 2 * length - 1))
    extended_tensor[...,:length] = tensor
    extended_tensor[...,length:] = torch.flip(tensor[...,1:], 
                                              dims=[-1])

    return make_circulant(extended_tensor, dim)[...,:length,:length]
