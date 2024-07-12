import torch
import numpy as np
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch

from models.hetero_gnn import BipartiteHeteroGNN
from utils.benchmark import sync_timer


class BaseBipartiteHeteroGNN(BipartiteHeteroGNN):
    def forward(self, data):
        batch_dict = data.batch_dict
        edge_index_dict, edge_attr_dict = data.edge_index_dict, data.edge_attr_dict

        # the only difference is, not to encode x start position
        x_dict = {'cons': self.b_encoder(data.b[:, None]),
                  'vals': self.obj_encoder(data.c[:, None])}

        hiddens = []
        for i in range(self.num_layers):
            h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict, batch_dict)
            keys = h2.keys()
            hiddens.append(h2['vals'])
            x_dict = {k: F.relu(h2[k]) for k in keys}

        x = self.predictor(hiddens[-1])
        return x, data.x_solution[:, None]

    @torch.no_grad()
    def evaluation(self, data, return_intern=False):
        obj_gaps = []
        time_steps = []
        opt_obj = data.obj_solution
        batched_c, _ = to_dense_batch(data.c, data['vals'].batch)  # batchsize x max_nnodes

        # prediction
        if return_intern:
            t_start = sync_timer()
        pred_x = self.forward(data)[0].squeeze(1)
        if return_intern:
            t_end = sync_timer()

        pred_x, _ = to_dense_batch(pred_x, data['vals'].batch)  # batchsize x max_nnodes
        batch_obj = (pred_x * batched_c).sum(1)
        if return_intern:
            obj_gaps.append(batch_obj)
            time_steps.append(t_end - t_start)

        if obj_gaps:
            obj_gaps = torch.abs((opt_obj - torch.cat(obj_gaps, dim=0)) / opt_obj).cpu().numpy()
            time_steps = np.cumsum(time_steps, axis=0)

        return torch.abs((opt_obj - batch_obj) / opt_obj), obj_gaps, time_steps
