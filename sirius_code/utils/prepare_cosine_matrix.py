import torch
import torch.nn as nn
import numpy as np

def prepare_cosine_matrix(order):

    # f_ij = cos(2 pi (i - 1)(j - 1) / order

    vec1 = torch.arange(order)
    vec2 = torch.arange(order)

    return torch.cos(2 * torch.tensor([np.pi]) * torch.outer(vec1, vec2) / order)
