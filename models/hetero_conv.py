from typing import Tuple

import torch
from torch import Tensor


class BipartiteConv(torch.nn.Module):
    def __init__(
            self,
            v2c_conv: torch.nn.Module,
            c2v_conv: torch.nn.Module,
    ):
        super().__init__()
        self.v2c_conv = v2c_conv
        self.c2v_conv = c2v_conv

    def forward(
            self,
            cons_embedding: torch.FloatTensor,
            vals_embedding: torch.FloatTensor,
            v2c_edge_index: torch.LongTensor,
            c2v_edge_index: torch.LongTensor,
            v2c_edge_attr: torch.FloatTensor,
            c2v_edge_attr: torch.FloatTensor,
            cons_batch: torch.LongTensor,
            vals_batch: torch.LongTensor) -> Tuple[Tensor, Tensor]:
        # update cons first
        cons_embedding = self.v2c_conv((vals_embedding, cons_embedding), v2c_edge_index, v2c_edge_attr, cons_batch)
        # update vals then
        vals_embedding = self.c2v_conv((cons_embedding, vals_embedding), c2v_edge_index, c2v_edge_attr, vals_batch)
        return vals_embedding, cons_embedding
