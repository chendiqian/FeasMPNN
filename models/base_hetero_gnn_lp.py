import torch
from models.base_hetero_conv_lp import BipartiteConvLP, TripartiteConvLP
from models.base_hetero_gnn import get_conv_layer, BipartiteHeteroGNN, TripartiteHeteroGNN


class BipartiteHeteroGNNLP(BipartiteHeteroGNN):
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
        super().__init__(conv,
                         head,
                         concat,
                         hid_dim,
                         num_encode_layers,
                         num_conv_layers,
                         num_pred_layers,
                         hid_pred,
                         num_mlp_layers,
                         norm,
                         plain_xstarts,
                         encode_start_x)

        del self.gcns
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gcns.append(BipartiteConvLP(
                get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
                get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
            ))


class TripartiteHeteroGNNLP(TripartiteHeteroGNN):
    def __init__(self, conv,
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
        super().__init__(conv,
                         head,
                         concat,
                         hid_dim,
                         num_encode_layers,
                         num_conv_layers,
                         num_pred_layers,
                         hid_pred,
                         num_mlp_layers,
                         norm,
                         plain_xstarts,
                         encode_start_x)

        del self.gcns
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gcns.append(TripartiteConvLP(
                v2c_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
                c2v_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
                # 1 node only so no normalization
                v2o_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, None, head, concat),
                o2v_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
                c2o_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, None, head, concat),
                o2c_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm, head, concat),
            ))
