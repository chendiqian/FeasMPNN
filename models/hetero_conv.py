from typing import Tuple, Optional
import inspect

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
            cons_embedding_0: torch.FloatTensor,
            vals_embedding_0: torch.FloatTensor,
            v2c_edge_index: torch.LongTensor,
            c2v_edge_index: torch.LongTensor,
            v2c_edge_attr: torch.FloatTensor,
            c2v_edge_attr: torch.FloatTensor,
            cons_batch: torch.LongTensor,
            vals_batch: torch.LongTensor,
            edge_norm: Optional[torch.FloatTensor]) -> Tuple[Tensor, Tensor]:

        has_skip = 'x_0' in inspect.signature(self.v2c_conv.forward).parameters.keys()

        # update cons first
        args = [(vals_embedding, cons_embedding)] + \
               ([cons_embedding_0] if has_skip else []) + \
               [v2c_edge_index, v2c_edge_attr, cons_batch] + \
               ([edge_norm] if edge_norm is not None else [])
        cons_embedding = self.v2c_conv(*args)

        args = [(cons_embedding, vals_embedding)] + \
               ([vals_embedding_0] if has_skip else []) + \
               [c2v_edge_index, c2v_edge_attr, vals_batch] + \
               ([edge_norm] if edge_norm is not None else [])
        vals_embedding = self.c2v_conv(*args)
        return vals_embedding, cons_embedding
