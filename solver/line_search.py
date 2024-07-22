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
