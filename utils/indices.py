import torch
import numpy as np


def square_idx(n):
    narange = np.arange(n)
    rows = narange.repeat(n)
    cols = np.tile(narange, n)
    return rows, cols


def torch_square_idx(n, device='cpu'):
    narange = torch.arange(n, device=device)
    rows = torch.repeat_interleave(narange, n)
    cols = narange.repeat(n)
    return rows, cols
