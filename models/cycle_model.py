import torch
from torch_geometric.utils import to_dense_batch
from data.utils import l1_normalize, line_search


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
        self.tau = 0.01
        self.reset_tau()

    def reset_tau(self):
        self.tau = 0.01

    def forward(self, data):

        # reset
        self.reset_tau()
        pred_list = []
        label_list = []
        for i in range(self.num_steps):
            # prediction
            pred = self.gnn(data)
            pred_list.append(pred)

            label = l1_normalize(data.x_solution - data.x_start)
            label_list.append(label)

            # Todo: experimental, barrier function
            direction = pred.detach() + self.tau / (data.x_start + self.tau)
            self.tau = max(self.tau / 2., 1.e-4)

            # projection
            pred = (data.proj_matrix @ direction[:, None]).squeeze()
            # line search
            alpha = line_search(data.x_start, pred, 1.)
            # update
            data.x_start = data.x_start + alpha * pred

        pred_list = torch.stack(pred_list, 1)
        label_list = torch.stack(label_list, 1)
        return pred_list, label_list

    @torch.no_grad()
    def evaluation(self, data):
        # reset
        self.reset_tau()
        current_best_batched_x, _ = to_dense_batch(data.x_start.clone(), data['vals'].batch)  # batchsize x max_nnodes
        batched_c, _ = to_dense_batch(data.c, data['vals'].batch)  # batchsize x max_nnodes
        opt_obj = data.obj_solution
        current_best_obj = (current_best_batched_x * batched_c).sum(1)
        for i in range(self.num_eval_steps):
            # prediction
            pred = self.gnn(data)
            direction = pred + self.tau / (data.x_start + self.tau)
            self.tau = max(self.tau / 2., 1.e-4)

            # projection
            pred = (data.proj_matrix @ direction[:, None]).squeeze()
            # line search
            alpha = line_search(data.x_start, pred, 1.)
            # update
            data.x_start = data.x_start + alpha * pred
            current_batched_x, _ = to_dense_batch(data.x_start, data['vals'].batch)  # batchsize x max_nnodes
            current_obj = (current_batched_x * batched_c).sum(1)
            better_mask = current_obj < current_best_obj
            current_best_obj = torch.where(better_mask, current_obj, current_best_obj)
            current_best_batched_x = torch.where(better_mask[:, None], current_batched_x, current_best_batched_x)

        return torch.abs((opt_obj - current_best_obj) / opt_obj)
