import os
import time

import numpy as np
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf
from qpsolvers import solve_qp
from torch_geometric.utils import scatter
from torch_sparse import SparseTensor


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


def l1_normalize(x: torch.Tensor, dim=0):
    x = x / x.abs().max(dim=dim, keepdims=True).values + 1.e-7
    return x


def batch_l1_normalize(x: torch.Tensor, batch: torch.Tensor, dim=0):
    maxs = scatter(x.abs(), batch, dim=dim, reduce='max') + 1.e-7
    maxs = maxs[batch]
    x = x / maxs
    return x


def save_run_config(args: DictConfig):
    if args.ckpt:
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        prefix = f'{args.wandb.project}_{args.wandb.name}'
        exist_runs = [d for d in os.listdir('logs') if d.startswith(prefix)]
        log_folder_name = f'logs/{prefix}_exp{len(exist_runs)}'
        os.mkdir(log_folder_name)
        # with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
        #     yaml.dump(vars(args), outfile, default_flow_style=False)
        OmegaConf.save(args, os.path.join(log_folder_name, 'config.yaml'))
        return log_folder_name
    return None


def project_solution(pred, A, b):
    """
    project a solution to the feasible region: Ax = b, x >= 0
    """
    P = np.eye(A.shape[1]).astype(np.float64)
    q = np.zeros(A.shape[1]).astype(np.float64)
    G = -np.eye(A.shape[1]).astype(np.float64)
    h = pred.astype(np.float64)
    Amatrix = A.astype(np.float64)

    bias = b - A @ pred
    proj = solve_qp(P, q, G, h, Amatrix, bias.astype(np.float64), solver="cvxopt")
    pred = pred + proj
    return pred


# def qp_obj(x, S, q, batch):
#     part = S * x[:, None]
#     Q = part @ part.t()
#     return scatter(Q.sum(0) * 0.5 + q * x, batch, reduce='sum')


def qp_obj(x, data):
    x_batch = data['vals'].batch
    P_edge_index = data.edge_index_dict[('vals', 'to', 'vals')]
    P_weight = data.edge_attr_dict[('vals', 'to', 'vals')].squeeze()
    P_edge_slice = data._slice_dict[('vals', 'to', 'vals')]['edge_index'].to(x.device)
    q = data.q

    edge_batch = P_edge_slice[1:] - P_edge_slice[:-1]
    if x.dim() > 1:
        P_weight = P_weight[:, None]
        q = q[:, None]
    edge_batch = torch.arange(len(edge_batch), device=x.device).repeat_interleave(edge_batch)
    xQx = scatter(x[P_edge_index[0]] * x[P_edge_index[1]] * P_weight * 0.5, edge_batch, dim=0, reduce='sum')
    qx = scatter(x * q, x_batch, dim=0, reduce='sum')
    return xQx + qx


def recover_qp_from_data(data, dtype=np.float32):
    data = data.to('cpu')
    q = data.q.numpy().astype(dtype)
    b = data.b.numpy().astype(dtype)
    A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
                     col=data['cons', 'to', 'vals'].edge_index[1],
                     value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
                     sparse_sizes=(data['cons'].num_nodes, data['vals'].num_nodes),
                     is_sorted=True, trust_data=True).to_dense().numpy().astype(dtype)
    P = SparseTensor(row=data['vals', 'to', 'vals'].edge_index[0],
                     col=data['vals', 'to', 'vals'].edge_index[1],
                     value=data['vals', 'to', 'vals'].edge_attr.squeeze(),
                     sparse_sizes=(data['vals'].num_nodes, data['vals'].num_nodes),
                     is_sorted=True, trust_data=True).to_dense().numpy().astype(dtype)
    # todo: might vary
    lb = np.zeros(A.shape[1]).astype(dtype)

    G, h, ub = (None,) * 3
    return P, q, A, b, G, h, lb, ub
