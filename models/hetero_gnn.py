import torch
from torch_geometric.nn import MLP

from models.genconv import GENConv
from models.gcnconv import GCNConv
from models.ginconv import GINEConv
from models.gcn2conv import GCN2Conv
from models.gatconv import GATv2Conv
from models.sgcnconv import SGCNConv
from models.hetero_conv import BipartiteConv


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
    elif conv.lower() == 'sgcnconv':
        return SGCNConv(edge_dim=1,
                        hid_dim=hid_dim)
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
                get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
                get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat)
            ))

        self.predictor = MLP([hid_dim] * num_pred_layers + [1], norm=None)

    def forward(self, data):
        vals_batch: torch.LongTensor = data['vals'].batch
        cons_batch: torch.LongTensor = data['cons'].batch
        c2v_edge_index: torch.LongTensor = data['cons', 'to', 'vals'].edge_index
        v2c_edge_index: torch.LongTensor = data['vals', 'to', 'cons'].edge_index
        c2v_edge_attr: torch.FloatTensor = data['cons', 'to', 'vals'].edge_attr
        v2c_edge_attr: torch.FloatTensor = data['vals', 'to', 'cons'].edge_attr

        cons_embedding = self.b_encoder(data.b[:, None])
        vals_embedding = self.start_pos_encoder(data.x_start[:, None]) + self.obj_encoder(data.c[:, None])

        edge_norms = data.norm if hasattr(data, 'norm') else None

        cons_embedding_0 = cons_embedding
        vals_embedding_0 = vals_embedding
        for i in range(self.num_layers):
            vals_embedding, cons_embedding = self.gcns[i](cons_embedding,
                                                          vals_embedding,
                                                          cons_embedding_0,
                                                          vals_embedding_0,
                                                          v2c_edge_index,
                                                          c2v_edge_index,
                                                          v2c_edge_attr,
                                                          c2v_edge_attr,
                                                          cons_batch,
                                                          vals_batch,
                                                          edge_norms)

        x = self.predictor(vals_embedding)
        return x.squeeze()
