from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP

from models.genconv import GENConv
from models.gcnconv import GCNConv
from models.ginconv import GINEConv
from models.hetero_conv import HeteroConv


def strseq2rank(conv_sequence):
    if conv_sequence == 'parallel':
        c2v = v2c = v2o = o2v = c2o = o2c = 0
    elif conv_sequence == 'cvo':
        v2c = o2c = 0
        c2v = o2v = 1
        c2o = v2o = 2
    elif conv_sequence == 'vco':
        c2v = o2v = 0
        v2c = o2c = 1
        c2o = v2o = 2
    elif conv_sequence == 'ocv':
        c2o = v2o = 0
        v2c = o2c = 1
        c2v = o2v = 2
    elif conv_sequence == 'ovc':
        c2o = v2o = 0
        c2v = o2v = 1
        v2c = o2c = 2
    elif conv_sequence == 'voc':
        c2v = o2v = 0
        c2o = v2o = 1
        v2c = o2c = 2
    elif conv_sequence == 'cov':
        v2c = o2c = 0
        c2o = v2o = 1
        c2v = o2v = 2
    else:
        raise ValueError
    return c2v, v2c, v2o, o2v, c2o, o2c


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


class TripartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 dropout,
                 norm,
                 use_res,
                 conv_sequence='parallel'):
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_conv_layers
        self.use_res = use_res

        self.encoder = torch.nn.ModuleDict({'vals': MLP([-1, hid_dim, hid_dim], norm=norm),
                                            'cons': MLP([-1, hid_dim, hid_dim], norm=norm),
                                            'obj': MLP([-1, hid_dim, hid_dim], norm=None)})
        self.start_pos_encoder = MLP([-1, hid_dim, hid_dim], norm=None)

        c2v, v2c, v2o, o2v, c2o, o2c = strseq2rank(conv_sequence)
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gcns.append(
                HeteroConv({
                    ('cons', 'to', 'vals'): (get_conv_layer(conv, hid_dim, num_mlp_layers, norm), c2v),
                    ('vals', 'to', 'cons'): (get_conv_layer(conv, hid_dim, num_mlp_layers, norm), v2c),
                    ('vals', 'to', 'obj'): (get_conv_layer(conv, hid_dim, num_mlp_layers, None), v2o),
                    ('obj', 'to', 'vals'): (get_conv_layer(conv, hid_dim, num_mlp_layers, norm), o2v),
                    ('cons', 'to', 'obj'): (get_conv_layer(conv, hid_dim, num_mlp_layers, None), c2o),
                    ('obj', 'to', 'cons'): (get_conv_layer(conv, hid_dim, num_mlp_layers, norm), o2c),
                }, aggr='cat'))

        self.predictor = MLP([-1] + [hid_dim] * (num_pred_layers - 1) + [1], norm=None)

    def forward(self, data):
        batch_dict = data.batch_dict
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict

        x_dict['cons'] = self.encoder['cons'](x_dict['cons'])
        x_dict['vals'] = torch.cat([self.encoder['vals'](x_dict['vals']),
                                    self.start_pos_encoder(data.x_start[:, None])], dim=1)
        x_dict['obj'] = self.encoder['obj'](x_dict['obj'])

        hiddens = []
        for i in range(self.num_layers):
            h1 = x_dict
            h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict, batch_dict)
            keys = h2.keys()
            hiddens.append(h2['vals'])

            if i < self.num_layers - 1:
                if self.use_res:
                    x_dict = {k: (F.relu(h2[k]) + h1[k]) / 2 for k in keys}
                else:
                    x_dict = {k: F.relu(h2[k]) for k in keys}
                x_dict = {k: F.dropout(F.relu(x_dict[k]), p=self.dropout, training=self.training) for k in keys}

        x = self.predictor(hiddens[-1])
        return x
