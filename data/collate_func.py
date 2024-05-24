from typing import List

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from solver.line_search import batch_line_search


def collate_fn_lp(graphs: List[Data], device: torch.device):
    if len(graphs) == 1:
        g = graphs[0]
        proj_matrix = g.proj_matrix.reshape(g.x_solution.shape[0], g.x_solution.shape[0])
    else:
        # # old version: block diagonal
        # cumsum_nnodes = 0
        #
        # row_indices = []
        # col_indices = []
        # vals = [g.proj_matrix for g in graphs]
        # max_nnodes = max([g.x_solution.shape[0] for g in graphs])
        # _arange = np.arange(max_nnodes)[:, None].repeat(max_nnodes, 1)
        #
        # for g in graphs:
        #     nnodes = g.x_solution.shape[0]
        #     tmp_arange = _arange[:nnodes, :nnodes] + cumsum_nnodes
        #     row_idx = tmp_arange.reshape(-1)
        #     col_idx = tmp_arange.T.reshape(-1)
        #     cumsum_nnodes += nnodes
        #     row_indices.append(row_idx)
        #     col_indices.append(col_idx)
        #
        # row_indices = torch.from_numpy(np.concatenate(row_indices, axis=0))
        # col_indices = torch.from_numpy(np.concatenate(col_indices, axis=0))
        # vals = torch.cat(vals, dim=0)
        #
        # proj_matrix = SparseTensor(row=row_indices, col=col_indices, value=vals, is_sorted=True, trust_data=True)

        # new version: batch x nnodes x nnodes
        nnodes = [g.x_solution.shape[0] for g in graphs]
        max_nnodes = max(nnodes)
        proj_matrix = torch.zeros(len(graphs), max_nnodes, max_nnodes)
        for i in range(len(graphs)):
            g = graphs[i]
            proj_matrix[i, :nnodes[i], :nnodes[i]] = g.proj_matrix.reshape(nnodes[i], nnodes[i])

    new_batch = Batch.from_data_list(graphs,
                                     exclude_keys=['A_row', 'A_col', 'A_val', 'b',
                                                   'proj_matrix', 'proj_mat_shape'])
    # finish the half of symmetric edges
    flip_tensor = torch.tensor([1, 0])
    new_batch[('vals', 'to', 'cons')].edge_index = new_batch[('cons', 'to', 'vals')].edge_index[flip_tensor]
    new_batch[('vals', 'to', 'cons')].edge_attr = new_batch[('cons', 'to', 'vals')].edge_attr
    new_batch[('obj', 'to', 'vals')].edge_index = new_batch[('vals', 'to', 'obj')].edge_index[flip_tensor]
    new_batch[('obj', 'to', 'vals')].edge_attr = new_batch[('vals', 'to', 'obj')].edge_attr
    new_batch[('obj', 'to', 'cons')].edge_index = new_batch[('cons', 'to', 'obj')].edge_index[flip_tensor]
    new_batch[('obj', 'to', 'cons')].edge_attr = new_batch[('cons', 'to', 'obj')].edge_attr

    new_batch.proj_matrix = proj_matrix

    # perturb the initial feasible solution
    proj_matrix = proj_matrix.to(device, non_blocking=True)

    # # old, when proj_matrix is block diagonal
    # direction = (proj_matrix @ torch.randn(new_batch.x_solution.shape[0], 1, device=device)).squeeze()

    # new
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
    return new_batch


def collate_fn_lp_bi(graphs: List[Data], device: torch.device):
    # this is bipartite graph, so we don't need the obj
    # but setting things in exclude_keys does not work
    # just remove keys in the first data of the datalist, because
    # https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/collate.py#L56
    g = graphs[0]
    del g['obj'], g[('vals', 'to', 'obj')], g[('cons', 'to', 'obj')], g[('obj', 'to', 'cons')], g[('obj', 'to', 'vals')]

    if len(graphs) == 1:
        g = graphs[0]
        proj_matrix = g.proj_matrix.reshape(g.x_solution.shape[0], g.x_solution.shape[0])
    else:
        nnodes = [g.x_solution.shape[0] for g in graphs]
        max_nnodes = max(nnodes)
        proj_matrix = torch.zeros(len(graphs), max_nnodes, max_nnodes)
        for i in range(len(graphs)):
            g = graphs[i]
            proj_matrix[i, :nnodes[i], :nnodes[i]] = g.proj_matrix.reshape(nnodes[i], nnodes[i])

    new_batch = Batch.from_data_list(graphs,
                                     exclude_keys=['A_row', 'A_col', 'A_val',
                                                   'proj_matrix', 'proj_mat_shape'])
    # finish the half of symmetric edges
    flip_tensor = torch.tensor([1, 0])
    new_batch[('vals', 'to', 'cons')].edge_index = new_batch[('cons', 'to', 'vals')].edge_index[flip_tensor]
    new_batch[('vals', 'to', 'cons')].edge_attr = new_batch[('cons', 'to', 'vals')].edge_attr

    new_batch.proj_matrix = proj_matrix

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
    return new_batch
