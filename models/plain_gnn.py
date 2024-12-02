import torch

from data.utils import qp_obj
from models.base_hetero_gnn import BipartiteHeteroGNN, TripartiteHeteroGNN
from trainer import Trainer


class PlainBipartiteHeteroGNN(BipartiteHeteroGNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encode_start_x=False)

    @torch.no_grad()
    def evaluation(self, data):
        opt_obj = data.obj_solution

        # prediction, hard non-negative
        pred_x = self.forward(data).relu()  # (nnodes,)

        batch_obj = qp_obj(pred_x, data)
        batch_violation = Trainer.violate_per_batch(pred_x, data)
        return pred_x, torch.abs((opt_obj - batch_obj) / opt_obj), batch_violation


class PlainTripartiteHeteroGNN(TripartiteHeteroGNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encode_start_x=False)

    @torch.no_grad()
    def evaluation(self, data):
        opt_obj = data.obj_solution

        # prediction
        pred_x = self.forward(data).relu()  # (nnodes,)

        batch_obj = qp_obj(pred_x, data)
        batch_violation = Trainer.violate_per_batch(pred_x, data)
        return pred_x, torch.abs((opt_obj - batch_obj) / opt_obj), batch_violation
