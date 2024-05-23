import time

import numpy as np
import seaborn as sns
import torch


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
