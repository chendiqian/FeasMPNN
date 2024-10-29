import argparse

import numpy as np
import wandb
from tqdm import tqdm

from data.dataset import LPDataset
from data.utils import sync_timer, recover_qp_from_data
from qpsolvers import solve_qp


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', default=False, action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath)[-1000:]

    cvxopt_time = []
    osqp_time = []

    pbar = tqdm(dataset)
    for data in pbar:
        P, q, A, b, G, h, lb, ub = recover_qp_from_data(data)

        start_t = sync_timer()
        solution = solve_qp(P, q, G, h, A, b, lb=lb, ub=ub, solver="cvxopt")
        end_t = sync_timer()
        cvxopt_time.append(end_t - start_t)

        start_t = sync_timer()
        solution = solve_qp(P, q, G, h, A, b, lb=lb, ub=ub, solver="osqp")
        end_t = sync_timer()
        osqp_time.append(end_t - start_t)

        stat_dict = {'cvxopt': cvxopt_time[-1],
                     'osqp': osqp_time[-1]}

        wandb.log(stat_dict)
        pbar.set_postfix(stat_dict)

    stat_dict = {"cvxopt_mean": np.mean(cvxopt_time),
                 "cvxopt_std": np.std(cvxopt_time),
                 "osqp_mean": np.mean(osqp_time),
                 "osqp_std": np.std(osqp_time)}

    wandb.log(stat_dict)
