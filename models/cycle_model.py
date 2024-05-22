import numpy as np
import torch
from torch_geometric.utils import to_dense_batch
from data.utils import l1_normalize, batch_line_search
import time


class CycleGNN(torch.nn.Module):
    def __init__(self,
                 num_steps: int,
                 num_eval_steps: int,
                 gnn: torch.nn.Module):
        super().__init__()

        self.num_steps = num_steps
        self.num_eval_steps = num_eval_steps
        self.gnn = gnn
        # Todo: experimental, barrier method
        self.init_tau = 0.01
        self.step_alpha = 5.

    def forward(self, data):

        # reset
        tau = self.init_tau
        pred_list = []
        label_list = []
        for i in range(self.num_steps):
            # prediction
            pred = self.gnn(data)
            pred_list.append(pred)

            label = l1_normalize(data.x_solution - data.x_start)
            label_list.append(label)

            # Todo: experimental, barrier function
            direction = pred.detach() + tau / (data.x_start + tau)
            tau = max(tau / 2., 1.e-5)

            # projection
            pred = (data.proj_matrix @ direction[:, None]).squeeze()
            # line search
            alpha = batch_line_search(data.x_start, pred, data['vals'].batch, self.step_alpha) * 0.995
            # update
            data.x_start = data.x_start + alpha * pred

        pred_list = torch.stack(pred_list, 1)
        label_list = torch.stack(label_list, 1)
        return pred_list, label_list

    @torch.no_grad()
    def evaluation(self, data, return_intern=False):
        # reset
        obj_gaps = []
        time_steps = []
        tau = self.init_tau
        current_best_batched_x, _ = to_dense_batch(data.x_start.clone(), data['vals'].batch)  # batchsize x max_nnodes
        batched_c, _ = to_dense_batch(data.c, data['vals'].batch)  # batchsize x max_nnodes
        opt_obj = data.obj_solution
        current_best_obj = (current_best_batched_x * batched_c).sum(1)
        for i in range(self.num_eval_steps):
            # prediction
            if return_intern:
                t_start = time.time()
            pred = self.gnn(data)
            direction = pred + tau / (data.x_start + tau)
            tau = max(tau / 2., 1.e-5)

            # projection
            pred = (data.proj_matrix @ direction[:, None]).squeeze()
            # line search
            alpha = batch_line_search(data.x_start, pred, data['vals'].batch, self.step_alpha) * 0.995
            # update
            data.x_start = data.x_start + alpha * pred
            if return_intern:
                t_end = time.time()
            current_batched_x, _ = to_dense_batch(data.x_start, data['vals'].batch)  # batchsize x max_nnodes
            current_obj = (current_batched_x * batched_c).sum(1)
            better_mask = current_obj < current_best_obj
            current_best_obj = torch.where(better_mask, current_obj, current_best_obj)
            current_best_batched_x = torch.where(better_mask[:, None], current_batched_x, current_best_batched_x)
            if return_intern:
                obj_gaps.append(current_best_obj)
                time_steps.append(t_end - t_start)

        if obj_gaps:
            obj_gaps = torch.abs((opt_obj - torch.cat(obj_gaps, dim=0)) / opt_obj).cpu().numpy()
            time_steps = np.cumsum(time_steps, axis=0)

        return torch.abs((opt_obj - current_best_obj) / opt_obj), obj_gaps, time_steps
