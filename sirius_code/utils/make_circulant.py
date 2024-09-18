import torch

def make_circulant(tensor, dim=0):
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], 
                    dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,))
