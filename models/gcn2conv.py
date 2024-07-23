from math import log
import torch.nn.functional as F
import torch
from torch_geometric.nn import MessagePassing, MLP, Linear
from torch_geometric.utils import degree
from torch_geometric.nn.inits import glorot


class GCN2Conv(MessagePassing):
    def __init__(self, edge_dim, hid_dim, num_mlp_layers, norm,
                 alpha: float = 0.1, theta: float = None, layer: int = None, ):
        super(GCN2Conv, self).__init__(aggr='add')

        self.lin_src = Linear(hid_dim, hid_dim)
        self.lin_dst = Linear(hid_dim, hid_dim)
        self.lin_edge = Linear(edge_dim, hid_dim)
        self.mlp = MLP([hid_dim] * (num_mlp_layers + 1), norm=norm)
        self.weight = torch.nn.Parameter(torch.empty(hid_dim, hid_dim))
        glorot(self.weight)

        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)

    def forward(self, x, x_0, edge_index, edge_attr, batch):
        x = (self.lin_src(x[0]), x[1])
        if edge_attr is not None and hasattr(self, 'lin_edge'):
            edge_attr = self.lin_edge(edge_attr)

        row, col = edge_index
        deg_src = degree(row, x[0].shape[0], dtype=x[0].dtype) + 1
        deg_src_inv_sqrt = deg_src.pow(-0.5)
        deg_src_inv_sqrt[deg_src_inv_sqrt == float('inf')] = 0

        deg_dst = degree(col, x[1].shape[0], dtype=x[1].dtype) + 1
        deg_dst_inv_sqrt = deg_dst.pow(-0.5)
        deg_dst_inv_sqrt[deg_dst_inv_sqrt == float('inf')] = 0

        norm = deg_src_inv_sqrt[row] * deg_dst_inv_sqrt[col]

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

        x_dst = x[1]
        x_dst = self.lin_dst(x_dst)
        out = out + x_dst

        out.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0

        out = out.add_(x_0)
        out = torch.addmm(out, out, self.weight, beta=1. - self.beta,
                          alpha=self.beta)

        return self.mlp(out, batch)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
