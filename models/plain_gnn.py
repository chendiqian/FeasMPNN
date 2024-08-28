import torch
import numpy as np
from torch_geometric.utils import to_dense_batch

from models.hetero_gnn import BipartiteHeteroGNN
from data.utils import sync_timer, batch_l1_normalize, project_solution
from solver.line_search import batch_line_search

from torch_sparse import SparseTensor


class BaseBipartiteHeteroGNN(BipartiteHeteroGNN):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs, encode_start_x=False)

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
        pred_x, real_node_mask = to_dense_batch(pred_x, data['vals'].batch)  # batchsize x max_nnodes
        batch_obj = (pred_x * batched_c).sum(1)
        return pred_x[real_node_mask], torch.abs((opt_obj - batch_obj) / opt_obj), None, None

    @torch.no_grad()
    def cycle_eval(self, data, num_eval_steps):
        """
        apply this to cycle model, but with a few modification
        """
        # one shot prediction
        t_start = sync_timer()
        pred_x = self.forward(data)[0]
        device = pred_x.device
        pred_x = pred_x.squeeze().detach().cpu().numpy()
        b = data.b.cpu().numpy()
        A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
                         col=data['cons', 'to', 'vals'].edge_index[1],
                         value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
                         is_sorted=True, trust_data=True).cpu().to_dense().numpy()
        # proj into feasible space
        pred_x = project_solution(pred_x, A, b)
        pred_x = torch.from_numpy(pred_x).float().to(device)
        time_total = sync_timer() - t_start

        # reset
        obj_gaps = []
        tau = 0.01
        step_alpha = 5.
        current_best_x = pred_x.clone()
        opt_obj = data.obj_solution[0]
        current_best_obj = current_best_x.dot(data.c)

        batch = data['vals'].batch
        for i in range(num_eval_steps):
            # prediction
            t_start = sync_timer()
            direction = pred_x - data.x_start
            direction = batch_l1_normalize(direction, batch)
            direction = direction + 3 * tau / (data.x_start + tau)
            tau = max(tau / 2., 1.e-5)

            # projection
            pred = torch.einsum('mn,n->m', data.proj_matrix, direction)

            # line search
            alpha = batch_line_search(data.x_start, pred, batch, step_alpha) * 0.995
            # update
            data.x_start = data.x_start + alpha * pred
            t_end = sync_timer()
            time_total = time_total + t_end - t_start

            current_obj = data.x_start.dot(data.c)
            if current_obj < current_best_obj:
                current_best_obj = current_obj
                current_best_x = data.x_start

            obj_gaps.append(current_best_obj)

        obj_gaps = torch.abs((opt_obj - torch.tensor(obj_gaps)) / opt_obj).cpu().numpy()
        project_x_objgap = torch.abs((opt_obj - pred_x.dot(data.c)) / opt_obj).cpu().numpy()
        return project_x_objgap, current_best_x, obj_gaps, time_total
