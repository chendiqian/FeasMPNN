import numpy as np
import torch
from torch_geometric.utils import to_dense_batch
from data.utils import sync_timer, qp_obj
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
        P_edge_index = data.edge_index_dict[('vals', 'to', 'vals')]
        P_weight = data.edge_attr_dict[('vals', 'to', 'vals')].squeeze()
        P_edge_slice = data._slice_dict[('vals', 'to', 'vals')]['edge_index'].to(x_start.device)
        current_obj = qp_obj(current_best_x, P_edge_index, P_weight, data.q, P_edge_slice, vals_batch)
        current_best_rel_obj_gap = torch.abs((opt_obj - current_obj) / opt_obj)

        for i in range(self.num_val_steps):
            # prediction
            t_start = sync_timer()
            pred = self.gnn(data, x_start).relu()
            t_end = sync_timer()

            # update
            x_start = pred

            current_obj = qp_obj(x_start, P_edge_index, P_weight, data.q, P_edge_slice, vals_batch)
            # unlike our method, here we DON'T have strict feasible solution, we use the absolute value of the error

            current_rel_obj_gap = torch.abs((opt_obj - current_obj) / opt_obj)

            better_mask = current_rel_obj_gap < current_best_rel_obj_gap
            current_best_rel_obj_gap = torch.where(better_mask, current_rel_obj_gap, current_best_rel_obj_gap)
            better_mask = better_mask[vals_batch]
            current_best_x = torch.where(better_mask, x_start, current_best_x)
            time_steps.append(t_end - t_start)

        time_steps = np.cumsum(time_steps, axis=0)

        return current_best_x, current_best_rel_obj_gap, time_steps
