import torch
from torch_geometric.utils import scatter


def l1_normalize(x: torch.Tensor, dim=0):
    x /= x.abs().max(dim=dim, keepdims=True).values + 1.e-7
    return x


def batch_l1_normalize(x: torch.Tensor, batch: torch.Tensor, dim=0):
    maxs = scatter(x, batch, dim=dim, reduce='max') + 1.e-7
    maxs = maxs[batch]
    x /= maxs
    return x
