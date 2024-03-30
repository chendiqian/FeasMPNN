import torch


class CycleGNN(torch.nn.Module):
    def __init__(self,
                 num_steps: int,
                 gnn: torch.nn.Module):
        super().__init__()

        self.num_steps = num_steps
        self.gnn = gnn

    def forward(self, data):
        pred_list = []
        for i in range(self.num_steps):
            # prediction
            pred = self.gnn(data)

            # projection
            pred = (data.proj_matrix @ pred).squeeze()

            alpha = 1.
            # line search
            neg_mask = pred.detach() < 0.
            if torch.any(neg_mask):
                alpha = min(alpha, (data.x_start[neg_mask] / -pred.detach()[neg_mask]).min().item())

            # update
            x_updated = data.x_start + alpha * pred

            pred_list.append(data.x_solution - x_updated)

            data.x_start = x_updated.detach()

        pred_list = torch.stack(pred_list, 1)
        return pred_list
