import os
import time

import numpy as np
import seaborn as sns
import torch
import yaml
from torch_geometric.utils import scatter


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


def save_run_config(args):
    if args.ckpt:
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        exist_runs = [d for d in os.listdir('logs') if d.startswith(args.wandbname)]
        log_folder_name = f'logs/{args.wandbname}exp{len(exist_runs)}'
        os.mkdir(log_folder_name)
        with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
            yaml.dump(vars(args), outfile, default_flow_style=False)
        return log_folder_name
    return None
