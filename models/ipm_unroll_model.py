from typing import Union

import numpy as np
import torch

from data.utils import sync_timer, qp_obj
from models.base_hetero_gnn import TripartiteHeteroGNN, BipartiteHeteroGNN

from trainer import Trainer


class IPMUnrollGNN(torch.nn.Module):
    def __init__(self,
                 num_steps: int,
                 num_val_steps: int,
                 gnn: Union[BipartiteHeteroGNN, TripartiteHeteroGNN]):
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
        opt_obj = data.obj_solution
        current_obj = qp_obj(x_start, data)
        current_obj_gap = torch.abs((opt_obj - current_obj) / opt_obj)
        current_vio = Trainer.violate_per_batch(x_start, data)

        objgaps = [current_obj_gap]
        vios = [current_vio]

        for i in range(self.num_val_steps):
            # prediction
            t_start = sync_timer()
            pred = self.gnn(data, x_start).relu()
            t_end = sync_timer()

            # update
            x_start = pred

            current_obj = qp_obj(x_start, data)
            current_obj_gap = torch.abs((opt_obj - current_obj) / opt_obj)
            objgaps.append(current_obj_gap)

            current_vio = Trainer.violate_per_batch(x_start, data)
            vios.append(current_vio)

            time_steps.append(t_end - t_start)

        time_steps = np.cumsum(time_steps, axis=0)

        vios = torch.stack(vios, dim=0)  # steps x batchsize
        objgaps = torch.stack(objgaps, dim=0)   # steps x batchsize
        thresh = torch.topk(vios, k=self.num_val_steps // 4, dim=0, largest=False, sorted=True).values[-1:, :]
        objgaps = torch.where(vios <= thresh, objgaps, 1.e5)
        best_objgaps = objgaps.min(0).values.float()
        vios = vios[objgaps.argmin(0), torch.arange(vios.shape[1])]

        return vios, best_objgaps, time_steps
