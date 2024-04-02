import torch
from data.utils import l1_normalize


class CycleGNN(torch.nn.Module):
    def __init__(self,
                 num_steps: int,
                 gnn: torch.nn.Module):
        super().__init__()

        self.num_steps = num_steps
        self.gnn = gnn

    def forward(self, data):
        pred_list = []
        label_list = []
        for i in range(self.num_steps):
            # prediction
            pred = self.gnn(data)
            pred_list.append(pred)

            label = l1_normalize(data.x_solution - data.x_start)
            label_list.append(label)

            # projection
            pred = (data.proj_matrix @ pred.detach()[:, None]).squeeze()

            alpha = 1.
            # line search
            neg_mask = pred < 0.
            if torch.any(neg_mask):
                alpha = min(alpha, (data.x_start[neg_mask] / -pred[neg_mask]).min().item())

            # update
            data.x_start = data.x_start + alpha * pred

        pred_list = torch.stack(pred_list, 1)
        label_list = torch.stack(label_list, 1)
        return pred_list, label_list
