from typing import Dict, List
import random
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch
from torch_geometric.data.hetero_data import to_homogeneous_edge_index
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from solver.linprog_ip import _ip_hsd
from solver.customized_solver import ipm_overleaf


def args_set_bool(args: Dict):
    for k, v in args.items():
        if isinstance(v, str):
            if v.lower() == 'true':
                args[k] = True
            elif v.lower() == 'false':
                args[k] = False
    return args


def collate_fn_ip(graphs: List[Data]):
    new_batch = Batch.from_data_list(graphs)
    row_bias = torch.hstack([new_batch.A_num_row.new_zeros(1), new_batch.A_num_row[:-1]]).cumsum(dim=0)
    row_bias = torch.repeat_interleave(row_bias, new_batch.A_nnz)
    new_batch.A_row += row_bias
    col_bias = torch.hstack([new_batch.A_num_col.new_zeros(1), new_batch.A_num_col[:-1]]).cumsum(dim=0)
    col_bias = torch.repeat_interleave(col_bias, new_batch.A_nnz)
    new_batch.A_col += col_bias
    return new_batch


def random_start_point(graph: Data, maxiter: int):
    A = SparseTensor(row=graph.A_row,
                     col=graph.A_col,
                     value=graph.A_val, is_sorted=True,
                     trust_data=True).to_dense().numpy()
    b = graph.b.numpy()
    c = graph.c.numpy()

    sol = ipm_overleaf(c, A, b, 'rand', 'cho', random.randint(1, maxiter))
    x, l, s = sol['x'], sol['lambd'], sol['s']

    # x, status, message, iteration, callback_outputs = _ip_hsd(A, b, c, 0.,
    #                                                           alpha0=0.99995, beta=0.1,
    #                                                           maxiter=random.randint(1, maxiter),
    #                                                           disp=False, tol=1.e-6, sparse=False,
    #                                                           lstsq=False, sym_pos=True, cholesky=None,
    #                                                           pc=True, ip=True, permc_spec='MMD_AT_PLUS_A',
    #                                                           callback=None,
    #                                                           postsolve_args=None,
    #                                                           rand_start=True)
    # x = x[:graph.c.shape[0]]

    graph.x_start = torch.from_numpy(x).to(torch.float)
    graph.l_start = torch.from_numpy(l).to(torch.float)
    graph.s_start = torch.from_numpy(s).to(torch.float)

    # direction
    x_direction = graph.x_solution - graph.x_start
    graph.x_label = x_direction / x_direction.abs().max() + 1.e-7
    l_direction = graph.l_solution - graph.l_start
    graph.l_label = l_direction / l_direction.abs().max() + 1.e-7
    s_direction = graph.s_solution - graph.s_start
    graph.s_label = s_direction / s_direction.abs().max() + 1.e-7

    # remove a, b, c unnecessary
    graph.A_row = None
    graph.A_col = None
    graph.A_val = None
    graph.b = None
    graph.c = None
    return graph


class HeteroAddLaplacianEigenvectorPE:
    def __init__(self, k, attr_name='laplacian_eigenvector_pe'):
        self.k = k
        self.attr_name = attr_name

    def __call__(self, data):
        if self.k == 0:
            return data
        data_homo = data.to_homogeneous()
        del data_homo.edge_weight
        lap = AddLaplacianEigenvectorPE(k=self.k, attr_name=self.attr_name)(data_homo).laplacian_eigenvector_pe

        _, node_slices, _ = to_homogeneous_edge_index(data)
        cons_lap = lap[node_slices['cons'][0]: node_slices['cons'][1], :]
        cons_lap = (cons_lap - cons_lap.mean(0)) / cons_lap.std(0)
        vals_lap = lap[node_slices['vals'][0]: node_slices['vals'][1], :]
        vals_lap = (vals_lap - vals_lap.mean(0)) / vals_lap.std(0)
        obj_lap = lap[node_slices['obj'][0]: node_slices['obj'][1], :]

        data['cons'].laplacian_eigenvector_pe = cons_lap
        data['vals'].laplacian_eigenvector_pe = vals_lap
        data['obj'].laplacian_eigenvector_pe = obj_lap
        return data
