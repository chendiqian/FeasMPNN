import numpy as np
import torch
from torch_geometric.utils import to_dense_batch
from data.utils import sync_timer, batch_l1_normalize
from solver.line_search import batch_line_search
from trainer import Trainer


class CycleGNN(torch.nn.Module):
    def __init__(self,
                 num_steps: int,
                 num_eval_steps: int,
                 gnn: torch.nn.Module,
                 init_tau: float,
                 tau_scale: float):
        super().__init__()

        self.num_steps = num_steps
        self.num_eval_steps = num_eval_steps
        self.gnn = gnn
        # Todo: experimental, barrier method
        self.init_tau = init_tau
        self.tau_scale = tau_scale
        self.step_alpha = 5.

    def forward(self, data):

        # this set of param generally works well for training
        tau = 0.01
        scale = 0.5
        pred_list = []
        label_list = []

        vals_batch = data['vals'].batch
        for i in range(self.num_steps):
            # prediction
            pred = self.gnn(data, data.x_start)
            pred_list.append(pred)

            label = batch_l1_normalize(data.x_solution - data.x_start, vals_batch)
            label_list.append(label)

            pred = batch_l1_normalize(pred.detach(), vals_batch)

            # barrier function
            direction = pred + 3 * tau / (data.x_start + tau)
            tau = max(tau * scale, 1.e-5)

            # projection
            if data.proj_matrix.dim() == 2:  # only 1 graph
                pred = torch.einsum('mn,n->m', data.proj_matrix, direction)
            else:
                direction, nmask = to_dense_batch(direction, vals_batch)  # batchsize x Nmax
                direction = torch.einsum('bnf,bn->bf', data.proj_matrix, direction)  # batchsize x Nmax x Neigs
                pred = torch.einsum('bnf,bf->bn', data.proj_matrix, direction)[nmask]

            # line search
            alpha = batch_line_search(data.x_start, pred, vals_batch, self.step_alpha) * 0.995
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
        x_start = data.x_start.clone()
        current_best_batched_x, real_node_mask = to_dense_batch(x_start, data['vals'].batch)  # batchsize x max_nnodes
        batched_c, _ = to_dense_batch(data.c, data['vals'].batch)  # batchsize x max_nnodes
        opt_obj = data.obj_solution
        current_best_obj = (current_best_batched_x * batched_c).sum(1)

        x_starts = []
        preds = []

        vals_batch = data['vals'].batch
        for i in range(self.num_eval_steps):
            x_starts.append(x_start)

            # prediction
            if return_intern:
                t_start = sync_timer()

            pred = self.gnn(data, x_start)
            pred = batch_l1_normalize(pred, vals_batch)
            preds.append(pred)
            direction = pred + 3. * tau / (x_start + tau)
            tau = max(tau * self.tau_scale, 1.e-5)

            # projection
            if data.proj_matrix.dim() == 2:  # only 1 graph
                pred = torch.einsum('mn,n->m', data.proj_matrix, direction)
            else:
                direction, nmask = to_dense_batch(direction, vals_batch)  # batchsize x Nmax
                direction = torch.einsum('bnf,bn->bf', data.proj_matrix, direction)  # batchsize x Nmax x Neigs
                pred = torch.einsum('bnf,bf->bn', data.proj_matrix, direction)[nmask]

            # line search
            alpha = batch_line_search(x_start, pred, vals_batch, self.step_alpha) * 0.995
            # update
            x_start = x_start + alpha * pred
            if return_intern:
                t_end = sync_timer()
            current_batched_x, _ = to_dense_batch(x_start, vals_batch)  # batchsize x max_nnodes
            current_obj = (current_batched_x * batched_c).sum(1)
            better_mask = current_obj < current_best_obj
            current_best_obj = torch.where(better_mask, current_obj, current_best_obj)
            current_best_batched_x = torch.where(better_mask[:, None], current_batched_x, current_best_batched_x)
            if return_intern:
                obj_gaps.append(current_best_obj)
                time_steps.append(t_end - t_start)

        if obj_gaps:
            obj_gaps = torch.stack(obj_gaps, dim=1)   # batchsize x steps
            obj_gaps = torch.abs((opt_obj[:, None] - obj_gaps) / opt_obj[:, None]).cpu().numpy()
            time_steps = np.cumsum(time_steps, axis=0)

        x_starts = torch.stack(x_starts, 1)  # nnodes x step
        preds = torch.stack(preds, 1)  # nnodes x step
        labels = data.x_solution[:, None] - x_starts
        cos_sims = 1. - Trainer.get_cos_sim(preds, labels, vals_batch)

        final_x = current_best_batched_x[real_node_mask]
        best_obj = torch.abs((opt_obj - current_best_obj) / opt_obj)
        return final_x, best_obj, obj_gaps, time_steps, cos_sims
