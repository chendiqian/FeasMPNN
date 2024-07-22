from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP

from models.genconv import GENConv
from models.gcnconv import GCNConv
from models.ginconv import GINEConv
from models.hetero_conv import HeteroConv


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
        self.b_encoder = MLP([-1, hid_dim, hid_dim], norm=None)
        self.start_pos_encoder = MLP([-1, hid_dim, hid_dim], norm=None)
        self.obj_encoder = MLP([-1, hid_dim, hid_dim], norm=None)

        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gcns.append(
                HeteroConv({
                    ('cons', 'to', 'vals'): (get_conv_layer(conv, hid_dim, num_mlp_layers, norm), 1),
                    ('vals', 'to', 'cons'): (get_conv_layer(conv, hid_dim, num_mlp_layers, norm), 0),
                }))

        self.predictor = MLP([-1] + [hid_dim] * (num_pred_layers - 1) + [1], norm=None)

    def forward(self, batch_dict, edge_index_dict, edge_attr_dict, b, c, x_start):

        x_dict = {'cons': self.b_encoder(b[:, None]),
                  'vals': self.start_pos_encoder(x_start[:, None]) + self.obj_encoder(c[:, None])}

        hiddens = []
        for i in range(self.num_layers):
            h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict, batch_dict)
            keys = h2.keys()
            hiddens.append(h2['vals'])
            x_dict = {k: F.relu(h2[k]) for k in keys}

        x = self.predictor(hiddens[-1])
        return x.squeeze()
