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
        edge_index = data[('cons', 'to', 'vals')].edge_index
        row, col = edge_index
        deg_src = degree(row, data['cons'].num_nodes, dtype=torch.float) + 1.
        deg_src_inv_sqrt = deg_src.pow(-0.5)
        deg_src_inv_sqrt[deg_src_inv_sqrt == float('inf')] = 0
        deg_dst = degree(col, data['vals'].num_nodes, dtype=torch.float) + 1.
        deg_dst_inv_sqrt = deg_dst.pow(-0.5)
        deg_dst_inv_sqrt[deg_dst_inv_sqrt == float('inf')] = 0
        norm = deg_src_inv_sqrt[row] * deg_dst_inv_sqrt[col]
        data.norm = norm
        return data
