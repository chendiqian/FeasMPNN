import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


class GCNNorm(BaseTransform):
    # adapted from
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/gcn_norm.html#GCNNorm
    def __init__(self):
        pass

    def forward(self, data: HeteroData) -> HeteroData:
        norm_dict = {}
        # for src, rel, dst in [('cons', 'to', 'vals'), ('vals', 'to', 'hids')]:
        for src, rel, dst in [('cons', 'to', 'vals'), ('vals', 'to', 'vals')]:
            edge_index = data[(src, rel, dst)].edge_index
            row, col = edge_index
            deg_src = degree(row, data[src].num_nodes, dtype=torch.float) + 1.
            deg_src_inv_sqrt = deg_src.pow(-0.5)
            deg_src_inv_sqrt[deg_src_inv_sqrt == float('inf')] = 0
            deg_dst = degree(col, data[dst].num_nodes, dtype=torch.float) + 1.
            deg_dst_inv_sqrt = deg_dst.pow(-0.5)
            deg_dst_inv_sqrt[deg_dst_inv_sqrt == float('inf')] = 0
            norm = deg_src_inv_sqrt[row] * deg_dst_inv_sqrt[col]
            norm_dict[(src, rel, dst)] = norm
            if src != dst:
                norm_dict[(dst, rel, src)] = norm
        data.norm_dict = norm_dict
        return data
