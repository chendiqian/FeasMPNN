from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import Linear, MLP
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax


class GATv2Conv(MessagePassing):
    def __init__(
        self,
        edge_dim: int,
        hid_dim: int,
        num_mlp_layers: int,
        norm: str,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.out_channels = hid_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = Linear(hid_dim, heads * hid_dim, weight_initializer='glorot')
        self.lin_r = Linear(hid_dim, heads * hid_dim, weight_initializer='glorot')
        # self.lin_dst = Linear(hid_dim, hid_dim, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, hid_dim))
        glorot(self.att)

        self.lin_edge = Linear(edge_dim, heads * hid_dim, bias=False, weight_initializer='glorot')
        # self.lin_edge_att = Linear(edge_dim, heads * hid_dim, bias=False, weight_initializer='glorot')
        self.mlp = MLP([-1] + [hid_dim] * num_mlp_layers, norm=norm, plain_last=False)

    def forward(self,
                x: Tuple[torch.FloatTensor, torch.FloatTensor],
                edge_index: torch.LongTensor,
                edge_attr: torch.FloatTensor,
                batch: torch.LongTensor):
        H, C = self.heads, self.out_channels

        x_l, x_r = x[0], x[1]
        x_l = self.lin_l(x_l).view(-1, H, C)
        x_r = self.lin_r(x_r).view(-1, H, C)

        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)
        # scatter_sum(alpha, edge_index[1], reduce='sum', dim=0) == all ones

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha, edge_attr=edge_attr)
        if self.concat:
            out = (out + x_r).reshape(out.shape[0], H * C)
        else:
            out = (out + x_r).mean(dim=1)
        return self.mlp(out, batch)

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        """
        x_j: src, edge indexed
        x_i: dst, edge indexed
        index: dst edge_index
        ptr: ptr for output index
        dim_size: target num_nodes
        """
        x = x_j + x_i

        # edge_attr = self.lin_edge_att(edge_attr)
        # edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
        x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)  # nedges x nheads x F
        alpha = (x * self.att).sum(dim=-1)  # nedges x nheads
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor, edge_attr: torch.FloatTensor) -> Tensor:
        return F.relu(x_j + edge_attr) * alpha.unsqueeze(-1)
