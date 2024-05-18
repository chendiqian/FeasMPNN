from typing import Dict, List
import time
import torch
from torch_sparse import SparseTensor
from torch_scatter import scatter_min
from torch_geometric.data import Data, Batch
import numpy as np
import seaborn as sns


def args_set_bool(args: Dict):
    for k, v in args.items():
        if isinstance(v, str):
            if v.lower() == 'true':
                args[k] = True
            elif v.lower() == 'false':
                args[k] = False
    return args


def collate_fn_lp(graphs: List[Data], device: torch.device):
    cumsum_nnodes = 0

    row_indices = []
    col_indices = []
    vals = [g.proj_matrix for g in graphs]
    max_nnodes = max([g.x_solution.shape[0] for g in graphs])
    _arange = np.arange(max_nnodes)[:, None].repeat(max_nnodes, 1)

    for g in graphs:
        nnodes = g.x_solution.shape[0]
        tmp_arange = _arange[:nnodes, :nnodes] + cumsum_nnodes
        row_idx = tmp_arange.reshape(-1)
        col_idx = tmp_arange.T.reshape(-1)
        cumsum_nnodes += nnodes
        row_indices.append(row_idx)
        col_indices.append(col_idx)

    row_indices = torch.from_numpy(np.concatenate(row_indices, axis=0))
    col_indices = torch.from_numpy(np.concatenate(col_indices, axis=0))
    vals = torch.cat(vals, dim=0)

    proj_matrix = SparseTensor(row=row_indices, col=col_indices, value=vals, is_sorted=True, trust_data=True)
    new_batch = Batch.from_data_list(graphs,
                                     exclude_keys=['A_row', 'A_col', 'A_val', 'b',
                                                   'proj_matrix', 'proj_mat_shape'])
    # finish the half of symmetric edges
    flip_tensor = torch.tensor([1, 0])
    new_batch[('vals', 'to', 'cons')].edge_index = new_batch[('cons', 'to', 'vals')].edge_index[flip_tensor]
    new_batch[('vals', 'to', 'cons')].edge_attr = new_batch[('cons', 'to', 'vals')].edge_attr
    new_batch[('obj', 'to', 'vals')].edge_index = new_batch[('vals', 'to', 'obj')].edge_index[flip_tensor]
    new_batch[('obj', 'to', 'vals')].edge_attr = new_batch[('vals', 'to', 'obj')].edge_attr
    new_batch[('obj', 'to', 'cons')].edge_index = new_batch[('cons', 'to', 'obj')].edge_index[flip_tensor]
    new_batch[('obj', 'to', 'cons')].edge_attr = new_batch[('cons', 'to', 'obj')].edge_attr

    new_batch.proj_matrix = proj_matrix

    # perturb the initial feasible solution
    proj_matrix = proj_matrix.to(device, non_blocking=True)
    direction = (proj_matrix @ torch.randn(new_batch.x_solution.shape[0], 1, device=device)).squeeze()
    alpha = batch_line_search(new_batch.x_feasible.to(device, non_blocking=True),
                              direction,
                              new_batch['vals'].batch.to(device, non_blocking=True),
                              1.)

    alpha = torch.rand(len(graphs), device=device)[new_batch['vals'].batch] * alpha
    new_batch.x_start = new_batch.x_feasible.to(device, non_blocking=True) + alpha * direction
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


def sync_timer():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def gaussian_filter(xs_grid, xs_data, ys_data, sigma):
    weights = np.exp(-(xs_data - xs_grid[:, None])**2 / (2 * sigma**2))
    return np.sum(weights * ys_data, 1) / np.sum(weights, 1)


def gaussian_filter_bt(xs_grid, xs_data, ys_data, sigma, n_boot=2000):
    bootstrap_res = sns.algorithms.bootstrap(
            np.column_stack((xs_data, ys_data)),
            func=lambda x: gaussian_filter(xs_grid, x[:, 0], x[:, 1], sigma=sigma),
            n_boot=n_boot)

    mean = bootstrap_res.mean(0)
    ci = sns.utils.ci(bootstrap_res, axis=0)
    return mean, ci
