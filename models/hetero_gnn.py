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
                   use_norm: bool):
    if conv.lower() == 'genconv':
        def get_conv():
            return GENConv(in_channels=-1,
                           out_channels=hid_dim,
                           num_layers=num_mlp_layers,
                           aggr='softmax',
                           msg_norm=use_norm,
                           learn_msg_scale=use_norm,
                           norm='batch' if use_norm else None,
                           bias=True,
                           edge_dim=1)
    elif conv.lower() == 'gcnconv':
        def get_conv():
            return GCNConv(edge_dim=1,
                           hid_dim=hid_dim,
                           num_mlp_layers=num_mlp_layers,
                           norm='batch' if use_norm else None)
    elif conv.lower() == 'ginconv':
        def get_conv():
            return GINEConv(edge_dim=1,
                            hid_dim=hid_dim,
                            num_mlp_layers=num_mlp_layers,
                            norm='batch' if use_norm else None)
    else:
        raise NotImplementedError

    return get_conv


class TripartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 pe_dim,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 dropout,
                 share_conv_weight,
                 use_norm,
                 use_res,
                 conv_sequence='parallel'):
        super().__init__()

        self.dropout = dropout
        self.share_conv_weight = share_conv_weight
        self.num_layers = num_conv_layers
        self.use_res = use_res

        if pe_dim > 0:
            self.pe_encoder = torch.nn.ModuleDict({
                'vals': MLP([-1, hid_dim, hid_dim]),
                'cons': MLP([-1, hid_dim, hid_dim]),
                'obj': MLP([-1, hid_dim, hid_dim])})
            in_emb_dim = hid_dim
        else:
            self.pe_encoder = None
            in_emb_dim = 2 * hid_dim

        self.encoder = torch.nn.ModuleDict({'vals': MLP([-1, hid_dim, in_emb_dim], norm='batch'),
                                            'cons': MLP([-1, hid_dim, in_emb_dim], norm='batch'),
                                            'obj': MLP([-1, hid_dim, in_emb_dim], norm='batch')})
        self.start_pos_encoder = MLP([-1, hid_dim, in_emb_dim], norm=None)  # shouldn't use batchnorm imo

        c2v, v2c, v2o, o2v, c2o, o2c = strseq2rank(conv_sequence)
        get_conv = get_conv_layer(conv, hid_dim, num_mlp_layers, use_norm)
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            if layer == 0 or not share_conv_weight:
                self.gcns.append(
                    HeteroConv({
                        ('cons', 'to', 'vals'): (get_conv(), c2v),
                        ('vals', 'to', 'cons'): (get_conv(), v2c),
                        ('vals', 'to', 'obj'): (get_conv(), v2o),
                        ('obj', 'to', 'vals'): (get_conv(), o2v),
                        ('cons', 'to', 'obj'): (get_conv(), c2o),
                        ('obj', 'to', 'cons'): (get_conv(), o2c),
                    }, aggr='cat'))

        self.pred_vals = MLP([-1] + [hid_dim] * (num_pred_layers - 1) + [1])
        # self.pred_cons = MLP([-1] + [hid_dim] * (num_pred_layers - 1) + [1])

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'vals', 'obj']:
            x_emb = self.encoder[k](x_dict[k])
            if self.pe_encoder is not None and hasattr(data[k], 'laplacian_eigenvector_pe'):
                pe_emb = 0.5 * (self.pe_encoder[k](data[k].laplacian_eigenvector_pe) +
                                self.pe_encoder[k](-data[k].laplacian_eigenvector_pe))
                x_emb = torch.cat([x_emb, pe_emb], dim=1)
            if k == 'vals' and hasattr(data, 'start_point'):
                start_pos = self.start_pos_encoder(data.start_point[:, None])
                x_emb = torch.cat([x_emb, start_pos], dim=1)
            x_dict[k] = x_emb

        hiddens = []
        for i in range(self.num_layers):
            if self.share_conv_weight:
                i = 0

            h1 = x_dict
            h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict)
            keys = h2.keys()
            hiddens.append((h2['cons'], h2['vals']))
            if self.use_res:
                h = {k: (F.relu(h2[k]) + h1[k]) / 2 for k in keys}
            else:
                h = {k: F.relu(h2[k]) for k in keys}
            h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in keys}
            x_dict = h

        # cons, vals = zip(*hiddens)
        vals = hiddens[-1][1]
        vals = self.pred_vals(vals)  # vals * 1
        return vals.squeeze()
