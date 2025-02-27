from typing import Dict, Optional

import torch
from torch_geometric.nn import MLP
from torch_geometric.typing import EdgeType, NodeType

from models.convs.genconv import GENConv
from models.convs.gcnconv import GCNConv
from models.convs.ginconv import GINEConv
from models.convs.gcn2conv import GCN2Conv
from models.convs.gatconv import GATv2Conv
from models.base_hetero_conv import BipartiteConv, TripartiteConv
from models.nn_utils import LogEncoder


def get_conv_layer(conv: str,
                   hid_dim: int,
                   num_mlp_layers: int,
                   norm: str,
                   head: int,
                   concat: bool):
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
    elif conv.lower() == 'gcn2conv':
        return GCN2Conv(edge_dim=1,
                        hid_dim=hid_dim,
                        num_mlp_layers=num_mlp_layers,
                        norm=norm)
    elif conv.lower() == 'gatconv':
        return GATv2Conv(edge_dim=1,
                         hid_dim=hid_dim,
                         num_mlp_layers=num_mlp_layers,
                         norm=norm,
                         heads=head,
                         concat=concat)
    else:
        raise NotImplementedError


class BipartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 head,
                 concat,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 hid_pred,
                 num_mlp_layers,
                 norm,
                 plain_xstarts=False,
                 encode_start_x=True):
        super().__init__()

        self.plain_xstarts = plain_xstarts
        self.num_layers = num_conv_layers
        self.b_encoder = MLP([1] + [hid_dim] * num_encode_layers, norm=None)
        if encode_start_x:
            self.start_pos_encoder = MLP([1] + [hid_dim] * num_encode_layers, norm=None)
            if not plain_xstarts:
                self.start_pos_encoder = torch.nn.Sequential(
                    LogEncoder(),
                    self.start_pos_encoder
                )
        else:
            self.start_pos_encoder = None
        self.q_encoder = MLP([1] + [hid_dim] * num_encode_layers, norm=None)

        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gcns.append(BipartiteConv(
                get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
                get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
                get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
            ))
        if hid_pred == -1:
            hid_pred = hid_dim
        self.predictor = MLP([hid_dim] + [hid_pred] * num_pred_layers + [1], norm=None)

    def init_embedding(self, data, x_start):
        batch_dict: Dict[NodeType, torch.LongTensor] = data.batch_dict
        edge_index_dict: Dict[EdgeType, torch.LongTensor] = data.edge_index_dict
        edge_attr_dict: Dict[EdgeType, torch.FloatTensor] = data.edge_attr_dict
        norm_dict: Dict[EdgeType, Optional[torch.FloatTensor]] = data.norm_dict

        cons_embedding = self.b_encoder(data.b[:, None])
        vals_embedding = self.q_encoder(data.q[:, None])
        if x_start is not None and self.start_pos_encoder is not None:
            vals_embedding = vals_embedding + self.start_pos_encoder(x_start[:, None])

        x_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding,
                                                     'cons': cons_embedding}
        x0_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding,
                                                      'cons': cons_embedding}
        return batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict, x0_dict

    def forward(self, data, x_start=None):
        batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict, x0_dict = self.init_embedding(data, x_start)

        for i in range(self.num_layers):
            x_dict = self.gcns[i](x_dict, x0_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        x = self.predictor(x_dict['vals'])
        return x.squeeze()


class TripartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 head,
                 concat,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 hid_pred,
                 num_mlp_layers,
                 norm,
                 plain_xstarts=True,
                 encode_start_x=True):
        super().__init__()

        self.plain_xstarts = plain_xstarts
        self.num_layers = num_conv_layers
        self.b_encoder = MLP([1] + [hid_dim] * num_encode_layers, norm=None)
        if encode_start_x:
            self.start_pos_encoder = MLP([1] + [hid_dim] * num_encode_layers, norm=None)
            if not plain_xstarts:
                self.start_pos_encoder = torch.nn.Sequential(
                    LogEncoder(),
                    self.start_pos_encoder
                )
        else:
            self.start_pos_encoder = None
        self.q_encoder = MLP([1] + [hid_dim] * num_encode_layers, norm=None)

        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gcns.append(TripartiteConv(
                v2v_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
                v2c_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
                c2v_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
                # 1 node only so no normalization
                v2o_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, None, head, concat),
                o2v_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
                c2o_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, None, head, concat),
                o2c_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
            ))
        if hid_pred == -1:
            hid_pred = hid_dim
        self.predictor = MLP([hid_dim] + [hid_pred] * num_pred_layers + [1], norm=None)

    def init_embedding(self, data, x_start):
        batch_dict: Dict[NodeType, torch.LongTensor] = data.batch_dict
        edge_index_dict: Dict[EdgeType, torch.LongTensor] = data.edge_index_dict
        edge_attr_dict: Dict[EdgeType, torch.FloatTensor] = data.edge_attr_dict
        norm_dict: Dict[EdgeType, Optional[torch.FloatTensor]] = data.norm_dict

        cons_embedding = self.b_encoder(data.b[:, None])
        vals_embedding = self.q_encoder(data.q[:, None])
        if x_start is not None and self.start_pos_encoder is not None:
            vals_embedding = vals_embedding + self.start_pos_encoder(x_start[:, None])

        x_dict: Dict[NodeType, torch.FloatTensor] = {
            'vals': vals_embedding,
            'cons': cons_embedding,
            # dumb initialization
            'obj': vals_embedding.new_zeros(data['obj'].num_nodes, vals_embedding.shape[1])}
        x0_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding,
                                                      'cons': cons_embedding,
                                                      'obj': x_dict['obj']}
        return batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict, x0_dict

    def forward(self, data, x_start=None):
        batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict, x0_dict = self.init_embedding(data, x_start)

        for i in range(self.num_layers):
            x_dict = self.gcns[i](x_dict, x0_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        x = self.predictor(x_dict['vals'])
        return x.squeeze()
