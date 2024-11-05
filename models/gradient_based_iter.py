import numpy as np
import torch
from torch_geometric.utils import to_dense_batch
from data.utils import sync_timer, qp_obj
from solver.line_search import batch_line_search
from torch.linalg import solve


class GradSolver(torch.nn.Module):
    def __init__(self,
                 num_eval_steps: int,
                 barrier_strength: float,
                 init_tau: float,
                 tau_scale: float):
        super().__init__()

        self.num_eval_steps = num_eval_steps
        self.init_tau = init_tau
        self.tau_scale = tau_scale
        self.barrier_strength = barrier_strength
        # if the pred is accurate, step length would be 1
        # we still do a line search, in case it violates positivity
        self.step_alpha = 1.

    @torch.no_grad()
    def evaluation(self, data, rhs):
        # reset
        obj_gaps = []
        time_steps = []
        tau = self.init_tau
        x_start = data.x_start.clone()
        current_best_x = x_start
        opt_obj = data.obj_solution
        vals_batch = data['vals'].batch
        P_edge_index = data.edge_index_dict[('vals', 'to', 'vals')]
        P_weight = data.edge_attr_dict[('vals', 'to', 'vals')].squeeze()
        P_edge_slice = data._slice_dict[('vals', 'to', 'vals')]['edge_index'].to(x_start.device)
        # current_best_obj = qp_obj(current_best_x, data.S, data.q, vals_batch)
        current_best_obj = qp_obj(current_best_x, P_edge_index, P_weight, data.q, P_edge_slice, vals_batch)

        for i in range(self.num_eval_steps):
            # prediction
            t_start = sync_timer()

            pred = - x_start + rhs
            # pred = batch_l1_normalize(pred, vals_batch)
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

            # current_obj = qp_obj(x_start, data.S, data.q, vals_batch)
            current_obj = qp_obj(x_start, P_edge_index, P_weight, data.q, P_edge_slice, vals_batch)
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

        best_obj = torch.abs((opt_obj - current_best_obj) / opt_obj)
        return current_best_x, best_obj, obj_gaps, time_steps
