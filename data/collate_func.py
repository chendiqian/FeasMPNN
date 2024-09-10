import types
from collections import defaultdict
from typing import List

import torch
from torch_geometric.data import Data, Batch
# from torch_sparse import SparseTensor


def collate_fn_lp_bi(graphs: List[Data]):
    # this is bipartite graph, so we don't need the obj
    # but setting things in exclude_keys does not work
    # just remove keys in the first data of the datalist, because
    # https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/collate.py#L56
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
    new_batch[('vals', 'to', 'cons')].edge_index = new_batch[('cons', 'to', 'vals')].edge_index[flip_tensor]
    new_batch[('vals', 'to', 'cons')].edge_attr = new_batch[('cons', 'to', 'vals')].edge_attr
    # new_batch[('hids', 'to', 'vals')].edge_index = new_batch[('vals', 'to', 'hids')].edge_index[flip_tensor]
    # new_batch[('hids', 'to', 'vals')].edge_attr = new_batch[('vals', 'to', 'hids')].edge_attr

    if not hasattr(new_batch, 'norm_dict'):
        new_batch.norm_dict = defaultdict(types.NoneType)

    # S_edge_index = [g[('vals', 'to', 'hids')].edge_index for g in graphs]
    # num_edges = [ed.shape[1] for ed in S_edge_index]
    # bias = new_batch['vals'].ptr[:-1]
    # bias = bias.repeat_interleave(torch.tensor(num_edges))
    # S_edge_index = torch.cat(S_edge_index, dim=1) + bias[None]
    # S = SparseTensor(row=S_edge_index[0],
    #                  col=S_edge_index[1],
    #                  value=new_batch[('vals', 'to', 'hids')].edge_attr.squeeze(),
    #                  sparse_sizes=(new_batch['vals'].num_nodes, new_batch['vals'].num_nodes),
    #                  is_sorted=True, trust_data=True)
    # new_batch.S = S
    return new_batch
