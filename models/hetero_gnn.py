from typing import Optional

import torch
from torch_geometric.nn import MLP

from models.genconv import GENConv
from models.gcnconv import GCNConv
from models.ginconv import GINEConv
from models.hetero_conv import BipartiteConv


def get_conv_layer(conv: str,
                   hid_dim: int,
                   num_mlp_layers: int,
                   norm: Optional[str]):
    if conv.lower() == 'genconv':
        return GENConv(in_channels=-1,
                       out_channels=hid_dim,
                       num_layers=num_mlp_layers,
                       aggr='softmax',
                       msg_norm=norm is not None,
                       learn_msg_scale=norm is not None,
                       norm=norm,
                       bias=True,
                       edge_dim=1)
    elif conv.lower() == 'gcnconv':
        return GCNConv(edge_dim=1,
                       hid_dim=hid_dim,
                       num_mlp_layers=num_mlp_layers,
                       norm=norm)
    elif conv.lower() == 'ginconv':
        return GINEConv(edge_dim=1,
                        hid_dim=hid_dim,
                        num_mlp_layers=num_mlp_layers,
                        norm=norm)
    else:
        raise NotImplementedError


class BipartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 norm):
        super().__init__()

        self.num_layers = num_conv_layers
        self.b_encoder = MLP([1, hid_dim, hid_dim], norm=None)
        self.start_pos_encoder = MLP([1, hid_dim, hid_dim], norm=None)
        self.obj_encoder = MLP([1, hid_dim, hid_dim], norm=None)

        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gcns.append(BipartiteConv(
                get_conv_layer(conv, hid_dim, num_mlp_layers, norm),
                get_conv_layer(conv, hid_dim, num_mlp_layers, norm)
            ))

        self.predictor = MLP([hid_dim] * num_pred_layers + [1], norm=None)

    def forward(self,
                v2c_edge_index: torch.LongTensor,
                c2v_edge_index: torch.LongTensor,
                v2c_edge_attr: torch.FloatTensor,
                c2v_edge_attr: torch.FloatTensor,
                cons_batch: torch.LongTensor,
                vals_batch: torch.LongTensor,
                b: torch.FloatTensor,
                c: torch.FloatTensor,
                x_start: torch.FloatTensor):

        cons_embedding = self.b_encoder(b[:, None])
        vals_embedding = self.start_pos_encoder(x_start[:, None]) + self.obj_encoder(c[:, None])

        for i in range(self.num_layers):
            vals_embedding, cons_embedding = self.gcns[i](cons_embedding,
                                                          vals_embedding,
                                                          v2c_edge_index,
                                                          c2v_edge_index,
                                                          v2c_edge_attr,
                                                          c2v_edge_attr,
                                                          cons_batch,
                                                          vals_batch)
            vals_embedding = torch.relu(vals_embedding)
            cons_embedding = torch.relu(cons_embedding)

        x = self.predictor(vals_embedding)
        return x.squeeze()
