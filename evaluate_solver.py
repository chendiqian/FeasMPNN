import argparse
import time

import cplex
import numpy as np
import wandb
from ortools.linear_solver import pywraplp
from torch_sparse import SparseTensor
from tqdm import tqdm

from data.dataset import LPDataset
from data.utils import sync_timer
from solver.linprog import linprog


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

    cplex_times = []
    sp_times = []
    scip_times = []

    for data in tqdm(dataset):
        c = data.c.numpy()
        b = data.b.numpy()
        A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
                         col=data['cons', 'to', 'vals'].edge_index[1],
                         value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
                         is_sorted=True, trust_data=True).to_dense().numpy()
        # opt_obj = data.obj_solution.item()

        # scipy
        start_t = sync_timer()
        res = linprog(
            c,
            A_ub=None, b_ub=None,
            A_eq=A, b_eq=b,
            bounds=None, method='interior-point')
        end_t = sync_timer()
        x = res.x
        obj = x.dot(c)
        sp_times.append(end_t - start_t)
        # sp_objgaps.append(np.abs((obj - opt_obj) / (opt_obj + 1.e-6)))

        # cplex
        num_decision_var = A.shape[1]
        num_constraints = A.shape[0]

        A = A.tolist()
        b = b.tolist()
        c = c.tolist()

        myProblem = cplex.Cplex()

        # Add the decision variables and set their lower bound and upper bound (if necessary)
        myProblem.variables.add(names=["x" + str(i) for i in range(num_decision_var)])
        for i in range(num_decision_var):
            myProblem.variables.set_lower_bounds(i, 0.0)
            myProblem.variables.set_upper_bounds(i, 1.0)

        # Add constraints
        for i in range(num_constraints):
            myProblem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[j for j in range(num_decision_var)], val=A[i])],
                rhs=[b[i]],
                names=["c" + str(i)],
                senses=["E"]
            )

        # Add objective function and set its sense
        for i in range(num_decision_var):
            myProblem.objective.set_linear([(i, c[i])])
        myProblem.objective.set_sense(myProblem.objective.sense.minimize)

        # Solve the model and print the answer
        t1 = time.time()
        myProblem.solve()
        t2 = time.time()
        x = myProblem.solution.get_values()
        obj = np.array(x).dot(np.array(c))

        cplex_times.append(end_t - start_t)
        # cplex_objgaps.append(np.abs(obj - opt_obj) / (opt_obj + 1.e-6))

        # ortools
        solver = pywraplp.Solver.CreateSolver('SCIP')

        x = [solver.IntVar(0, 1, f'x{i}') for i in range(num_decision_var)]

        objective = solver.Objective()
        for i in range(num_decision_var):
            objective.SetCoefficient(x[i], c[i])
        objective.SetMinimization()

        for i in range(num_constraints):
            solver.Add(sum([x[j] * A[i][j] for j in range(num_decision_var)]) == b[i])

        t1 = time.time()
        status = solver.Solve()
        t2 = time.time()
        scip_times.append(t2 - t1)
        x = np.array([x[i].solution_value() for i in range(num_decision_var)])

    stat_dict = {"scipy_mean": np.mean(sp_times),
                 "scipy_std": np.std(sp_times),
                 "cplex_mean": np.mean(cplex_times),
                 "cplex_std": np.std(cplex_times),
                 "scip_mean": np.mean(scip_times),
                 "scip_std": np.std(scip_times)}

    if args.use_wandb:
        wandb.log(stat_dict)
