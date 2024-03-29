from typing import Dict, List
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch
import numpy as np
from solver.linprog_ip import _ip_hsd
# from scipy.optimize import linprog
from scipy.sparse import block_diag


def args_set_bool(args: Dict):
    for k, v in args.items():
        if isinstance(v, str):
            if v.lower() == 'true':
                args[k] = True
            elif v.lower() == 'false':
                args[k] = False
    return args


def feasible_start_point(graph: Data):
    A = SparseTensor(row=graph.A_row,
                     col=graph.A_col,
                     value=graph.A_val, is_sorted=True,
                     trust_data=True).to_dense().numpy()
    b = graph.b.numpy()

    x, status, message, iteration, callback_outputs = _ip_hsd(A, b, np.zeros(A.shape[1]), 0.,
                                                              alpha0=0.99995, beta=0.1,
                                                              maxiter=10,
                                                              disp=False, tol=1.e-6, sparse=False,
                                                              lstsq=False, sym_pos=True, cholesky=None,
                                                              pc=True, ip=True, permc_spec='MMD_AT_PLUS_A',
                                                              callback=None,
                                                              postsolve_args=None,
                                                              rand_start=True)

    # # another option, faster on large instance
    # rn = np.random.rand(b.shape[0])
    # x1 = linprog(np.random.randn(A.shape[1]), A_eq=A, b_eq=b + rn, bounds=None).x
    # x2 = linprog(np.random.randn(A.shape[1]), A_eq=A, b_eq=b - rn, bounds=None).x
    # x = (x1 + x2) / 2

    graph.x_start = torch.from_numpy(x).to(torch.float)

    # remove a, b, c unnecessary
    graph.A_row = None
    graph.A_col = None
    graph.A_val = None
    graph.b = None
    graph.c = None
    return graph


def collate_fn_lp(graphs: List[Data]):
    proj_matrices = [g.pop('proj_matrix').numpy().reshape(g.pop('proj_mat_shape').tolist()) for g in graphs]
    new_batch = Batch.from_data_list(graphs)
    proj_matrices = block_diag(proj_matrices)
    proj_matrices = SparseTensor.from_scipy(proj_matrices)
    new_batch.proj_matrix = proj_matrices
    return new_batch
