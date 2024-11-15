import pdb

import torch
from torch_geometric.utils import scatter


def line_search(x: torch.Tensor, direction: torch.Tensor, alpha: float = 1.0):
    """
    line search to ensure x' = x + alpha * direction >= 0
    """
    neg_mask = direction < 0.
    if torch.any(neg_mask):
        alpha = min(alpha, (x[neg_mask] / -direction[neg_mask]).min().item())
    return alpha


def batch_line_search(x: torch.Tensor, direction: torch.Tensor, batch: torch.Tensor, alpha: float = 1.0):
    neg_mask = direction < 0.
    alpha = torch.where(neg_mask, x / -direction, alpha)
    alpha = scatter(alpha, batch, reduce='min')
    alpha = alpha[batch]
    return alpha


def convex_line_search(d, x, P_edge_index, P_weight, q, slice, x_batch, min_alpha, max_alpha):
    edge_batch = slice[1:] - slice[:-1]
    edge_batch = torch.arange(len(edge_batch), device=x.device).repeat_interleave(edge_batch)
    dQx = scatter(d[P_edge_index[0]] * x[P_edge_index[1]] * P_weight, edge_batch, dim=0, reduce='sum')
    dQd = scatter(d[P_edge_index[0]] * d[P_edge_index[1]] * P_weight, edge_batch, dim=0, reduce='sum')
    cd = scatter(q * d, x_batch, dim=0, reduce='sum')
    alpha = - (dQx + cd) / dQd
    alpha[alpha > max_alpha] = max_alpha
    alpha[alpha < min_alpha] = min_alpha
    alpha = alpha[x_batch]
    return alpha
