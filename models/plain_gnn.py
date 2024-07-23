import torch
import numpy as np
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch

from models.hetero_gnn import BipartiteHeteroGNN
from utils.benchmark import sync_timer
from utils.data import l1_normalize
from solver.line_search import batch_line_search


class BaseBipartiteHeteroGNN(BipartiteHeteroGNN):
    def forward(self, data):
        # the only difference is, not to encode x start position
        cons_embedding = self.b_encoder(data.b[:, None])
        vals_embedding = self.obj_encoder(data.c[:, None])

        edge_norms = data.norm if hasattr(data, 'norm') else None
        cons_embedding_0 = cons_embedding
        vals_embedding_0 = vals_embedding
        for i in range(self.num_layers):
            vals_embedding, cons_embedding = self.gcns[i](cons_embedding,
                                                          vals_embedding,
                                                          cons_embedding_0,
                                                          vals_embedding_0,
                                                          data['vals', 'to', 'cons'].edge_index,
                                                          data['cons', 'to', 'vals'].edge_index,
                                                          data['vals', 'to', 'cons'].edge_attr,
                                                          data['cons', 'to', 'vals'].edge_attr,
                                                          data['cons'].batch,
                                                          data['vals'].batch,
                                                          edge_norms)

        x = self.predictor(vals_embedding)
        if not self.training:
            x = torch.relu(x)  # hard non negative
        return x, data.x_solution[:, None]

    @torch.no_grad()
    def evaluation(self, data):
        opt_obj = data.obj_solution
        batched_c, _ = to_dense_batch(data.c, data['vals'].batch)  # batchsize x max_nnodes

        # prediction
        pred_x = self.forward(data)[0].squeeze(1)
        pred_x, _ = to_dense_batch(pred_x, data['vals'].batch)  # batchsize x max_nnodes
        batch_obj = (pred_x * batched_c).sum(1)
        return torch.abs((opt_obj - batch_obj) / opt_obj), None, None

    @torch.no_grad()
    def cycle_eval(self, data, num_eval_steps):
        """
        apply this to cycle model, but with a few modification
        """
        # reset
        obj_gaps = []
        time_steps = []
        tau = 0.01
        step_alpha = 5.
        current_best_batched_x, _ = to_dense_batch(data.x_start.clone(), data['vals'].batch)  # batchsize x max_nnodes
        batched_c, _ = to_dense_batch(data.c, data['vals'].batch)  # batchsize x max_nnodes
        opt_obj = data.obj_solution
        current_best_obj = (current_best_batched_x * batched_c).sum(1)

        batch = data['vals'].batch
        for i in range(num_eval_steps):
            # prediction
            t_start = sync_timer()
            pred_x = self.forward(data)[0].squeeze()
            direction = pred_x - data.x_start
            direction = l1_normalize(direction)
            direction = direction + tau / (data.x_start + tau)
            tau = max(tau / 2., 1.e-5)

            # projection
            if data.proj_matrix.dim() == 2:  # only 1 graph
                pred = torch.einsum('mn,n->m', data.proj_matrix, direction)
            else:
                direction, nmask = to_dense_batch(direction, batch)
                pred = torch.einsum('bnm,bm->bn', data.proj_matrix, direction)[nmask]

            # line search
            alpha = batch_line_search(data.x_start, pred, batch, step_alpha) * 0.995
            # update
            data.x_start = data.x_start + alpha * pred
            t_end = sync_timer()
            current_batched_x, _ = to_dense_batch(data.x_start, batch)  # batchsize x max_nnodes
            current_obj = (current_batched_x * batched_c).sum(1)
            better_mask = current_obj < current_best_obj
            current_best_obj = torch.where(better_mask, current_obj, current_best_obj)
            current_best_batched_x = torch.where(better_mask[:, None], current_batched_x, current_best_batched_x)

            obj_gaps.append(current_best_obj)
            time_steps.append(t_end - t_start)

        obj_gaps = torch.abs((opt_obj - torch.cat(obj_gaps, dim=0)) / opt_obj).cpu().numpy()
        time_steps = np.cumsum(time_steps, axis=0)

        return current_best_batched_x, torch.abs((opt_obj - current_best_obj) / opt_obj), obj_gaps, time_steps
