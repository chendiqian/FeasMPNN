import torch

from models.base_hetero_gnn_lp import BipartiteHeteroGNNLP, TripartiteHeteroGNNLP
from trainer import Trainer
from torch_scatter import scatter_sum


class PlainBipartiteHeteroGNNLP(BipartiteHeteroGNNLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encode_start_x=False)

    @torch.no_grad()
    def evaluation(self, data):
        opt_obj = data.obj_solution

        # prediction, hard non-negative
        pred_x = self.forward(data).relu()  # (nnodes,)

        batch_obj = scatter_sum(pred_x * data.q, data['vals'].batch, dim=0)
        batch_violation = Trainer.violate_per_batch(pred_x, data)
        return pred_x, torch.abs((opt_obj - batch_obj) / opt_obj), batch_violation


class PlainTripartiteHeteroGNNLP(TripartiteHeteroGNNLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encode_start_x=False)

    @torch.no_grad()
    def evaluation(self, data):
        opt_obj = data.obj_solution

        # prediction
        pred_x = self.forward(data).relu()  # (nnodes,)

        batch_obj = scatter_sum(pred_x * data.q, data['vals'].batch, dim=0)
        batch_violation = Trainer.violate_per_batch(pred_x, data)
        return pred_x, torch.abs((opt_obj - batch_obj) / opt_obj), batch_violation
