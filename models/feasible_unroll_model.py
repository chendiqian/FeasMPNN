from typing import Union

import numpy as np
import torch
from torch_geometric.utils import to_dense_batch

from data.utils import sync_timer, qp_obj, batch_l1_normalize
from models.base_hetero_gnn import BipartiteHeteroGNN, TripartiteHeteroGNN
from solver.line_search import batch_line_search
from trainer import Trainer
from torch_scatter import scatter_sum


class FeasibleUnrollGNN(torch.nn.Module):
    def __init__(self,
                 num_steps: int,
                 train_frac: float,
                 num_eval_steps: int,
                 gnn: Union[BipartiteHeteroGNN, TripartiteHeteroGNN],
                 barrier_strength: float,
                 init_tau: float,
                 tau_scale: float):
        super().__init__()

        self.num_steps = num_steps
        assert 0. < train_frac <= 1.
        assert int(train_frac * num_steps) > 0
        self.train_frac = train_frac
        self.num_eval_steps = num_eval_steps
        self.gnn = gnn
        # Todo: experimental, barrier method
        self.init_tau = init_tau
        self.tau_scale = tau_scale
        self.barrier_strength = barrier_strength
        # if the pred is accurate, step length would be 1
        # we still do a line search, in case it violates positivity
        self.step_alpha = 1.

    def forward(self, data):
        tau = self.init_tau
        scale = self.tau_scale
        pred_list = []
        label_list = []

        if self.train_frac < 1.:
            train_steps = np.random.choice(self.num_steps, int(self.train_frac * self.num_steps), replace=False)
        else:
            train_steps = np.arange(self.num_steps)

        vals_batch = data['vals'].batch
        for i in range(self.num_steps):
            # prediction
            if i in train_steps:
                pred = self.gnn(data, data.x_start)
            else:
                with torch.no_grad():
                    pred = self.gnn(data, data.x_start)

            pred_list.append(pred)
            label = data.x_solution - data.x_start
            label_list.append(label)
            pred = pred.detach()

            # barrier function
            # todo: tune this factor
            direction = pred + self.barrier_strength * tau / (data.x_start + tau)
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
    def evaluation(self, data):
        # reset
        obj_gaps = []
        time_steps = []
        tau = self.init_tau
        x_start = data.x_start.clone()
        current_best_x = x_start
        opt_obj = data.obj_solution
        vals_batch = data['vals'].batch
        current_best_obj = qp_obj(current_best_x, data)

        x_starts = []
        preds = []

        for i in range(self.num_eval_steps):
            x_starts.append(x_start)

            # prediction
            t_start = sync_timer()

            pred = self.gnn(data, x_start)
            # pred = batch_l1_normalize(pred, vals_batch)
            preds.append(pred)
            direction = pred + self.barrier_strength * tau / (x_start + tau)
            tau = max(tau * self.tau_scale, 1.e-5)

            # projection
            if data.proj_matrix.dim() == 2:  # only 1 graph
                pred = torch.einsum('mn,n->m', data.proj_matrix, direction)
            else:
                direction, nmask = to_dense_batch(direction, vals_batch)  # batchsize x Nmax
                direction = torch.einsum('bnf,bn->bf', data.proj_matrix, direction)  # batchsize x Nmax x Neigs
                pred = torch.einsum('bnf,bf->bn', data.proj_matrix, direction)[nmask]

            # min 0.5 (x + a * d)^T Q (x + a * d) + c^T(x + a * d)
            # => a = - (d^T Q x + c^T d) / (d^T Q d)
            # alpha1 = convex_line_search(
            #     pred, x_start, P_edge_index, P_weight, data.q, P_edge_slice, vals_batch, 0.5, 5.)

            # line search
            alpha = batch_line_search(x_start, pred, vals_batch, self.step_alpha) * 0.995
            # alpha = torch.min(alpha, alpha1)

            # update
            x_start = x_start + alpha * pred
            t_end = sync_timer()

            current_obj = qp_obj(x_start, data)
            # since we have strict feasible solution, we use the value obj
            better_mask = current_obj < current_best_obj
            current_best_obj = torch.where(better_mask, current_obj, current_best_obj)
            better_mask = better_mask[vals_batch]
            current_best_x = torch.where(better_mask, x_start, current_best_x)
            obj_gaps.append(current_best_obj)
            time_steps.append(t_end - t_start)

        obj_gaps = torch.stack(obj_gaps, dim=1)   # batchsize x steps
        obj_gaps = torch.abs((opt_obj[:, None] - obj_gaps) / opt_obj[:, None])
        time_steps = np.cumsum(time_steps, axis=0)

        x_starts = torch.stack(x_starts, 1)  # nnodes x step
        preds = torch.stack(preds, 1)  # nnodes x step
        labels = data.x_solution[:, None] - x_starts
        cos_sims = 1. - Trainer.get_cos_sim(preds, labels, vals_batch)

        best_obj = torch.abs((opt_obj - current_best_obj) / opt_obj)
        return current_best_x, best_obj, obj_gaps, time_steps, cos_sims


class FeasibleUnrollGNNLP(torch.nn.Module):
    def __init__(self,
                 num_steps: int,
                 num_eval_steps: int,
                 gnn: Union[BipartiteHeteroGNN, TripartiteHeteroGNN],
                 barrier_strength: float,
                 init_tau: float,
                 tau_scale: float):
        super().__init__()

        self.num_steps = num_steps
        self.num_eval_steps = num_eval_steps
        self.gnn = gnn
        # Todo: experimental, barrier method
        self.init_tau = init_tau
        self.tau_scale = tau_scale
        self.barrier_strength = barrier_strength
        # for LPs, the solution is always at the vertex,
        # so we always take a step as large as possible
        self.step_alpha = 10.

    def forward(self, data):
        tau = self.init_tau
        scale = self.tau_scale
        pred_list = []
        label_list = []

        vals_batch = data['vals'].batch
        for i in range(self.num_steps):
            # prediction
            pred = self.gnn(data, data.x_start)
            pred_list.append(pred)
            # unlike QP, we supervise on the normalized direction
            label = batch_l1_normalize(data.x_solution - data.x_start, vals_batch)
            label_list.append(label)

            pred = batch_l1_normalize(pred.detach(), vals_batch)

            # barrier function
            direction = pred + self.barrier_strength * tau / (data.x_start + tau)
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
    def evaluation(self, data):
        # reset
        obj_gaps = []
        time_steps = []
        tau = self.init_tau
        x_start = data.x_start.clone()
        current_best_x = x_start.clone()
        opt_obj = data.obj_solution
        vals_batch = data['vals'].batch
        current_best_obj = scatter_sum(x_start * data.q, vals_batch, dim=0)

        x_starts = []
        preds = []

        for i in range(self.num_eval_steps):
            x_starts.append(x_start)

            # prediction
            t_start = sync_timer()

            pred = self.gnn(data, x_start)
            pred = batch_l1_normalize(pred, vals_batch)
            preds.append(pred)
            direction = pred + self.barrier_strength * tau / (x_start + tau)
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
            t_end = sync_timer()

            current_obj = scatter_sum(x_start * data.q, vals_batch, dim=0)
            # since we have strict feasible solution, we use the value obj
            better_mask = current_obj < current_best_obj
            current_best_obj = torch.where(better_mask, current_obj, current_best_obj)
            better_mask = better_mask[vals_batch]
            current_best_x = torch.where(better_mask, x_start, current_best_x)
            obj_gaps.append(current_best_obj)
            time_steps.append(t_end - t_start)

        obj_gaps = torch.stack(obj_gaps, dim=1)   # batchsize x steps
        obj_gaps = torch.abs((opt_obj[:, None] - obj_gaps) / opt_obj[:, None])
        time_steps = np.cumsum(time_steps, axis=0)

        x_starts = torch.stack(x_starts, 1)  # nnodes x step
        preds = torch.stack(preds, 1)  # nnodes x step
        labels = data.x_solution[:, None] - x_starts
        cos_sims = 1. - Trainer.get_cos_sim(preds, labels, vals_batch)

        best_obj = torch.abs((opt_obj - current_best_obj) / opt_obj)
        return current_best_x, best_obj, obj_gaps, time_steps, cos_sims
