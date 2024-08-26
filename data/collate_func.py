from typing import List

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from solver.line_search import batch_line_search


def collate_fn_lp_bi(graphs: List[Data], perturb: bool = False, device: torch.device = 'cpu'):
    # this is bipartite graph, so we don't need the obj
    # but setting things in exclude_keys does not work
    # just remove keys in the first data of the datalist, because
    # https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/collate.py#L56
    g = graphs[0]
    del g['obj'], g[('vals', 'to', 'obj')], g[('cons', 'to', 'obj')], g[('obj', 'to', 'cons')], g[('obj', 'to', 'vals')]

    has_proj = hasattr(g, 'proj_matrix') and g.proj_matrix is not None
    if has_proj:
        assert g.proj_matrix.shape[0] == g.x_solution.shape[0] ** 2
    else:
        assert hasattr(g, 'nulls')

    if len(graphs) == 1:
        g = graphs[0]
        if has_proj:
            proj_matrix = g.proj_matrix.reshape(g.x_solution.shape[0], g.x_solution.shape[0])
        else:
            nulls = g.nulls.reshape(g.x_solution.shape[0], -1)
            proj_matrix = nulls @ nulls.t()
    else:
        nnodes = [g.x_solution.shape[0] for g in graphs]
        max_nnodes = max(nnodes)
        proj_matrix = torch.zeros(len(graphs), max_nnodes, max_nnodes)
        for i in range(len(graphs)):
            g = graphs[i]
            if has_proj:
                proj_matrix[i, :nnodes[i], :nnodes[i]] = g.proj_matrix.reshape(nnodes[i], nnodes[i])
            else:
                nulls = g.nulls.reshape(nnodes[i], -1)
                proj_matrix[i, :nnodes[i], :nulls.shape[1]] = nulls
        if not has_proj:
            proj_matrix = proj_matrix.to(device, non_blocking=True)
            proj_matrix = torch.einsum('bmf,bnf->bmn', proj_matrix, proj_matrix)

    new_batch = Batch.from_data_list(graphs,
                                     exclude_keys=['A_row', 'A_col', 'A_val', 'x',
                                                   'nulls', 'proj_matrix', 'proj_mat_shape'])
    # finish the half of symmetric edges
    flip_tensor = torch.tensor([1, 0])
    new_batch[('vals', 'to', 'cons')].edge_index = new_batch[('cons', 'to', 'vals')].edge_index[flip_tensor]
    new_batch[('vals', 'to', 'cons')].edge_attr = new_batch[('cons', 'to', 'vals')].edge_attr

    new_batch.proj_matrix = proj_matrix

    if perturb:
        # perturb the initial feasible solution
        proj_matrix = proj_matrix.to(device, non_blocking=True)
        batch = new_batch['vals'].batch.to(device, non_blocking=True)
        if len(graphs) == 1:
            direction = torch.randn(new_batch.x_solution.shape[0], 1, device=device)
            direction = (proj_matrix @ direction).squeeze()
        else:
            direction = torch.randn(new_batch.x_solution.shape[0], 1, device=device)
            direction, nmask = to_dense_batch(direction, batch)
            direction = torch.einsum('bnm,bmf->bnf', proj_matrix, direction)[nmask].squeeze()

        alpha = batch_line_search(new_batch.x_feasible.to(device, non_blocking=True),
                                  direction,
                                  batch,
                                  1.)

        alpha = torch.rand(len(graphs), device=device)[batch] * alpha
        new_batch.x_start = new_batch.x_feasible.to(device, non_blocking=True) + alpha * direction
    else:
        # use the fixed semi-random starting point
        new_batch.x_start = new_batch.x_feasible

    return new_batch


def collate_fn_lp_base(graphs: List[Data]):
    g = graphs[0]
    del g['obj'], g[('vals', 'to', 'obj')], g[('cons', 'to', 'obj')], g[('obj', 'to', 'cons')], g[('obj', 'to', 'vals')]
    del g.proj_matrix, g.nulls, g.x_feasible

    new_batch = Batch.from_data_list(graphs, exclude_keys=['x'])  # we drop the dumb x features
    # finish the half of symmetric edges
    flip_tensor = torch.tensor([1, 0])
    new_batch[('vals', 'to', 'cons')].edge_index = new_batch[('cons', 'to', 'vals')].edge_index[flip_tensor]
    new_batch[('vals', 'to', 'cons')].edge_attr = new_batch[('cons', 'to', 'vals')].edge_attr

    return new_batch
