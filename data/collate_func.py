from typing import List

import torch
from torch_geometric.data import Data, Batch


def collate_fn_lp_bi(graphs: List[Data]):
    if len(graphs) == 1:
        g = graphs[0]
        nulls = g.nulls.reshape(g.x_solution.shape[0], -1)
        proj_matrix = nulls @ nulls.t()
    else:
        nnodes = [g.x_solution.shape[0] for g in graphs]
        num_eigs = [g.nulls.shape[0] // g.x_solution.shape[0] for g in graphs]
        max_nnodes = max(nnodes)
        proj_matrix = torch.zeros(len(graphs), max_nnodes, max(num_eigs))
        for i in range(len(graphs)):
            g = graphs[i]
            nulls = g.nulls.reshape(nnodes[i], num_eigs[i])
            proj_matrix[i, :nnodes[i], :num_eigs[i]] = nulls

    new_batch = collate_fn_lp_base(graphs)
    new_batch.proj_matrix = proj_matrix
    # use the fixed semi-random starting point
    new_batch.x_start = new_batch.x_feasible
    return new_batch


def collate_fn_lp_base(graphs: List[Data]):
    new_batch = Batch.from_data_list(graphs, exclude_keys=['x', 'nulls'])  # we drop the dumb x features
    # finish the half of symmetric edges
    flip_tensor = torch.tensor([1, 0])
    for k, v in graphs[0].edge_index_dict.items():
        src, rel, dst = k
        new_batch[(dst, rel, src)].edge_index = new_batch[(src, rel, dst)].edge_index[flip_tensor]
        new_batch[(dst, rel, src)].edge_attr = new_batch[(src, rel, dst)].edge_attr

    if not hasattr(new_batch[('vals', 'to', 'cons')], 'norm'):
        norm_dict = {}
        for k, v in new_batch.edge_index_dict.items():
            norm_dict[k] = None
        new_batch.norm_dict = norm_dict

    return new_batch
