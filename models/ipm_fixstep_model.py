import torch

from data.utils import qp_obj
from models.base_hetero_gnn import TripartiteHeteroGNN, BipartiteHeteroGNN
from trainer import Trainer


class FixStepTripartiteIPMGNN(TripartiteHeteroGNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encode_start_x=False)

    def forward(self, data, last_layer=False):
        batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict, x0_dict = self.init_embedding(data, None)

        preds = []
        for i in range(self.num_layers):
            x_dict = self.gcns[i](x_dict, x0_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)
            preds.append(x_dict['vals'])

        if last_layer:
            preds = preds[-1]
        else:
            preds = torch.stack(preds, dim=1)  # vals * steps * feature
        preds = self.predictor(preds)
        return preds.squeeze(-1)

    @torch.no_grad()
    def evaluation(self, data):
        opt_obj = data.obj_solution

        # prediction
        pred_x = self.forward(data, True).relu()

        batch_obj = qp_obj(pred_x, data)
        batch_violation = Trainer.violate_per_batch(pred_x, data)
        return pred_x, torch.abs((opt_obj - batch_obj) / opt_obj), batch_violation


class FixStepBipartiteIPMGNN(BipartiteHeteroGNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encode_start_x=False)

    def forward(self, data, last_layer=False):
        batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict, x0_dict = self.init_embedding(data, None)

        preds = []
        for i in range(self.num_layers):
            x_dict = self.gcns[i](x_dict, x0_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)
            preds.append(x_dict['vals'])

        if last_layer:
            preds = preds[-1]
        else:
            preds = torch.stack(preds, dim=1)  # vals * steps * feature
        preds = self.predictor(preds)
        return preds.squeeze(-1)

    @torch.no_grad()
    def evaluation(self, data):
        opt_obj = data.obj_solution

        # prediction
        pred_x = self.forward(data, True).relu()

        batch_obj = qp_obj(pred_x, data)
        batch_violation = Trainer.violate_per_batch(pred_x, data)
        return pred_x, torch.abs((opt_obj - batch_obj) / opt_obj), batch_violation
