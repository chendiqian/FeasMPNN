import argparse

import numpy as np
import wandb
from torch_sparse import SparseTensor
from tqdm import tqdm

from data.dataset import LPDataset
from data.utils import sync_timer
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
        q = data.q.numpy().astype(np.float64)
        b = data.b.numpy().astype(np.float64)
        A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
                         col=data['cons', 'to', 'vals'].edge_index[1],
                         value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
                         is_sorted=True, trust_data=True).to_dense().numpy().astype(np.float64)
        P = SparseTensor(row=data['vals', 'to', 'vals'].edge_index[0],
                         col=data['vals', 'to', 'vals'].edge_index[1],
                         value=data['vals', 'to', 'vals'].edge_attr.squeeze(),
                         is_sorted=True, trust_data=True).to_dense().numpy().astype(np.float64)
        lb = np.zeros(A.shape[1]).astype(np.float64)

        start_t = sync_timer()
        solution = solve_qp(P, q, None, None, A, b, lb=lb, solver="cvxopt")
        end_t = sync_timer()
        cvxopt_time.append(end_t - start_t)

        start_t = sync_timer()
        solution = solve_qp(P, q, None, None, A, b, lb=lb, solver="osqp")
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
