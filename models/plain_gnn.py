from torch_geometric.utils import to_dense_batch
from typing import Dict, Optional

import torch
from torch_geometric.nn import MLP
from torch_geometric.typing import EdgeType, NodeType
from models.hetero_gnn import BipartiteHeteroGNN
from data.utils import sync_timer, batch_l1_normalize, qp_obj


class BaseBipartiteHeteroGNN(BipartiteHeteroGNN):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs, encode_start_x=False)

    def forward(self, data):
        batch_dict: Dict[NodeType, torch.LongTensor] = data.batch_dict
        edge_index_dict: Dict[EdgeType, torch.LongTensor] = data.edge_index_dict
        edge_attr_dict: Dict[EdgeType, torch.FloatTensor] = data.edge_attr_dict
        norm_dict: Dict[EdgeType, Optional[torch.FloatTensor]] = data.norm_dict

        cons_embedding = self.b_encoder(data.b[:, None])
        vals_embedding = self.q_encoder(data.q[:, None])

        x_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding,
                                                     'cons': cons_embedding}
        x0_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding,
                                                      'cons': cons_embedding}

        for i in range(self.num_layers):
            x_dict = self.gcns[i](x_dict, x0_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        x = self.predictor(x_dict['vals'])
        if not self.training:
            x = torch.relu(x)  # hard non negative
        return x, data.x_solution[:, None]

    @torch.no_grad()
    def evaluation(self, data):
        opt_obj = data.obj_solution

        # prediction
        pred_x = self.forward(data)[0].squeeze(1)

        vals_batch = data['vals'].batch
        P_edge_index = data.edge_index_dict[('vals', 'to', 'vals')]
        P_weight = data.edge_attr_dict[('vals', 'to', 'vals')].squeeze()
        P_edge_slice = data._slice_dict[('vals', 'to', 'vals')]['edge_index'].to(pred_x.device)

        batch_obj = qp_obj(pred_x, P_edge_index, P_weight, data.q, P_edge_slice, vals_batch)
        return pred_x, torch.abs((opt_obj - batch_obj) / opt_obj)
