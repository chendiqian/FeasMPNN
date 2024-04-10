from typing import Dict, List
import torch
from torch_sparse import SparseTensor
from torch_scatter import scatter_min
from torch_geometric.data import Data, Batch
import numpy as np


def args_set_bool(args: Dict):
    for k, v in args.items():
        if isinstance(v, str):
            if v.lower() == 'true':
                args[k] = True
            elif v.lower() == 'false':
                args[k] = False
    return args


def collate_fn_lp(graphs: List[Data], device: torch.device):
    cumsum_row = 0
    cumsum_col = 0

    row_indices = []
    col_indices = []
    vals = []

    for g in graphs:
        vals.append(g.pop('proj_matrix'))
        nrows, ncols = g.pop('proj_mat_shape').tolist()
        row_idx = np.arange(cumsum_row, cumsum_row + nrows).repeat(ncols)
        cumsum_row += nrows
        col_idx = np.tile(np.arange(cumsum_col, cumsum_col + ncols), nrows)
        cumsum_col += ncols
        row_indices.append(row_idx)
        col_indices.append(col_idx)

    row_indices = torch.from_numpy(np.concatenate(row_indices, axis=0))
    col_indices = torch.from_numpy(np.concatenate(col_indices, axis=0))
    vals = torch.cat(vals, dim=0)

    proj_matrix = SparseTensor(row=row_indices, col=col_indices, value=vals, is_sorted=True, trust_data=True)
    new_batch = Batch.from_data_list(graphs)
    new_batch.proj_matrix = proj_matrix

    # perturb the initial feasible solution
    proj_matrix = proj_matrix.to(device, non_blocking=True)
    direction = (proj_matrix @ torch.randn(new_batch.x_solution.shape[0], 1, device=device)).squeeze()
    alpha = batch_line_search(new_batch.x_feasible.to(device, non_blocking=True),
                              direction,
                              new_batch['vals'].batch.to(device, non_blocking=True),
                              1.)

    alpha = torch.rand(len(graphs), device=device)[new_batch['vals'].batch] * alpha
    new_batch.x_start = new_batch.x_feasible + alpha * direction
    return new_batch


def l1_normalize(x: torch.Tensor, dim=0):
    x /= x.abs().max(dim=dim, keepdims=True).values + 1.e-7
    return x


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
    alpha = scatter_min(alpha, batch)[0]
    alpha = alpha[batch]
    return alpha
