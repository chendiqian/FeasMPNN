import torch
from data.utils import l1_normalize


class CycleGNN(torch.nn.Module):
    def __init__(self,
                 num_steps: int,
                 gnn: torch.nn.Module):
        super().__init__()

        self.num_steps = num_steps
        self.gnn = gnn
        # Todo: experimental, barrier method
        self.tau = 0.01

    def forward(self, data):
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

            alpha = 1.
            # line search
            neg_mask = pred < 0.
            if torch.any(neg_mask):
                alpha = min(alpha, (data.x_start[neg_mask] / -pred[neg_mask]).min().item())

            # update
            data.x_start = data.x_start + alpha * pred

            # print(data.c.dot(data.x_start))

        pred_list = torch.stack(pred_list, 1)
        label_list = torch.stack(label_list, 1)
        return pred_list, label_list
