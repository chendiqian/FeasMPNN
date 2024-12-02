from typing import Dict, Optional

import numpy as np
import torch
from torch_geometric.typing import NodeType, EdgeType

from data.utils import sync_timer, qp_obj
from models.base_hetero_gnn import TripartiteHeteroGNN
from trainer import Trainer


class IPMGNN(torch.nn.Module):
    def __init__(self,
                 num_steps: int,
                 num_val_steps: int,
                 gnn: torch.nn.Module):
        super().__init__()
        self.num_steps = num_steps
        self.num_val_steps = num_val_steps
        self.gnn = gnn

    def forward(self, data):
        pred_list = []

        # the first point
        x_start = data.x_feasible
        for i in range(self.num_steps):
            # prediction
            pred = self.gnn(data, x_start)
            pred_list.append(pred)
            # todo: maybe use pred as the next point?
            x_start = data.trajectory[:, i]

        pred_list = torch.stack(pred_list, 1)
        return pred_list

    @torch.no_grad()
    def evaluation(self, data):
        # reset
        time_steps = []
        x_start = data.x_feasible.clone()
        current_best_x = x_start
        opt_obj = data.obj_solution
        vals_batch = data['vals'].batch
        current_obj = qp_obj(current_best_x, data)
        current_best_rel_obj_gap = torch.abs((opt_obj - current_obj) / opt_obj)

        for i in range(self.num_val_steps):
            # prediction
            t_start = sync_timer()
            pred = self.gnn(data, x_start).relu()
            t_end = sync_timer()

            # update
            x_start = pred

            current_obj = qp_obj(x_start, data)
            # unlike our method, here we DON'T have strict feasible solution, we use the absolute value of the error

            current_rel_obj_gap = torch.abs((opt_obj - current_obj) / opt_obj)

            better_mask = current_rel_obj_gap < current_best_rel_obj_gap
            current_best_rel_obj_gap = torch.where(better_mask, current_rel_obj_gap, current_best_rel_obj_gap)
            better_mask = better_mask[vals_batch]
            current_best_x = torch.where(better_mask, x_start, current_best_x)
            time_steps.append(t_end - t_start)

        time_steps = np.cumsum(time_steps, axis=0)

        return current_best_x, current_best_rel_obj_gap, time_steps


class FixStepIPMGNN(TripartiteHeteroGNN):
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
                 norm):
        super().__init__(conv,
                         head,
                         concat,
                         hid_dim,
                         num_encode_layers,
                         num_conv_layers,
                         num_pred_layers,
                         hid_pred,
                         num_mlp_layers,
                         norm, False, False)

    def forward(self, data, last_layer=False):
        batch_dict: Dict[NodeType, torch.LongTensor] = data.batch_dict
        edge_index_dict: Dict[EdgeType, torch.LongTensor] = data.edge_index_dict
        edge_attr_dict: Dict[EdgeType, torch.FloatTensor] = data.edge_attr_dict
        norm_dict: Dict[EdgeType, Optional[torch.FloatTensor]] = data.norm_dict

        cons_embedding = self.b_encoder(data.b[:, None])
        vals_embedding = self.q_encoder(data.q[:, None])

        x_dict: Dict[NodeType, torch.FloatTensor] = {
            'vals': vals_embedding,
            'cons': cons_embedding,
            # dumb initialization
            'obj': vals_embedding.new_zeros(data['obj'].num_nodes, vals_embedding.shape[1])}
        x0_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding,
                                                      'cons': cons_embedding,
                                                      'obj': x_dict['obj']}

        preds = []
        for i in range(self.num_layers):
            x_dict = self.gcns[i](x_dict, x0_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)
            preds.append(x_dict['vals'])

        if last_layer:
            preds = preds[-1]
        else:
            preds = torch.stack(preds, dim=1)  # vals * steps * feature
        preds = self.predictor(preds)
        return preds.squeeze(-1)

    @torch.no_grad()
    def evaluation(self, data):
        opt_obj = data.obj_solution

        # prediction
        pred_x = self.forward(data, True).relu()

        batch_obj = qp_obj(pred_x, data)
        batch_violation = Trainer.violate_per_batch(pred_x, data)
        return pred_x, torch.abs((opt_obj - batch_obj) / opt_obj), batch_violation
