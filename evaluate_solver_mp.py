import argparse
import time
import multiprocessing

import numpy as np
import wandb
from ortools.linear_solver import pywraplp
from torch_sparse import SparseTensor

from data.dataset import LPDataset



def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', default=False, action='store_true')
    return parser.parse_args()


def prepare_lp(data):
    c = data.c.numpy()
    b = data.b.numpy()
    A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
                     col=data['cons', 'to', 'vals'].edge_index[1],
                     value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
                     is_sorted=True, trust_data=True).to_dense().numpy()

    num_decision_var = A.shape[1]
    num_constraints = A.shape[0]

    A = A.tolist()
    b = b.tolist()
    c = c.tolist()

    solver = pywraplp.Solver.CreateSolver('SCIP')

    x = [solver.NumVar(0, 1, f'x{i}') for i in range(num_decision_var)]

    objective = solver.Objective()
    for i in range(num_decision_var):
        objective.SetCoefficient(x[i], c[i])
    objective.SetMinimization()

    for i in range(num_constraints):
        solver.Add(sum([x[j] * A[i][j] for j in range(num_decision_var)]) == b[i])
    return solver


def solve_lp(data):
    lp = prepare_lp(data)
    t1 = time.time()
    status = lp.Solve()
    t2 = time.time()
    return t2 - t1


if __name__ == '__main__':
    args = args_parser()

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath)[-1000:]

    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(
        processes=pool_size,
    )
    pool_outputs = pool.map(solve_lp, [g for g in dataset])
    pool.close()  # no more tasks
    pool.join()  # wrap up current tasks

    stat_dict = {"scip_mean": np.mean(pool_outputs), "scip_std": np.std(pool_outputs)}
    wandb.log(stat_dict)
