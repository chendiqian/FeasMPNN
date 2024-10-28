import numpy as np
import torch
from torch_geometric.utils import to_dense_batch, scatter
from torch_sparse import spmm
from data.utils import qp_obj

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cos_metric = torch.nn.CosineEmbeddingLoss(reduction='none')


class Trainer:
    def __init__(self,
                 loss_type,
                 microbatch,
                 coeff_l2,
                 coeff_cos,
                 ):
        # self.best_val_loss = 1.e8
        # self.best_cos_sim = 1.e8
        self.best_objgap = 1.e8
        self.patience = 0
        self.loss_type = loss_type
        if loss_type == 'l2':
            self.loss_func = lambda x: x ** 2
        elif loss_type == 'l1':
            self.loss_func = lambda x: x.abs()
        elif loss_type == 'cos':
            pass
        else:
            raise ValueError
        self.coeff_l2 = coeff_l2
        self.coeff_cos = coeff_cos
        self.microbatch = microbatch

    def train(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        train_losses = 0.
        cos_sims = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)

            pred, label = model(data)  # nnodes x steps
            loss = self.get_loss(pred, label, data['vals'].batch)
            cos_sim = self.get_cos_sim(pred, label, data['vals'].batch)  # batchsize x steps

            train_losses += loss.detach() * data.num_graphs
            cos_sims = cos_sims + cos_sim.detach().sum(0)
            num_graphs += data.num_graphs

            # use both L2 loss and Cos similarity loss
            loss = self.coeff_l2 * loss + self.coeff_cos * cos_sim.mean()
            loss = loss / self.microbatch
            loss.backward()
            if (i + 1) % self.microbatch == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()

        return (train_losses / num_graphs).item(), (cos_sims / num_graphs).cpu().tolist()

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        halfs = []
        lasts = []
        for i, data in enumerate(dataloader):
            data = data.to(device)
            _, best_obj_gap, obj_gaps, _, _ = model.evaluation(data)
            _, steps = obj_gaps.shape
            half_objgap = obj_gaps[:, steps // 2]
            last_objgap = obj_gaps[:, -1]
            halfs.append(half_objgap)
            lasts.append(last_objgap)

        halfs = torch.cat(halfs, dim=0).mean().item()
        lasts = torch.cat(lasts, dim=0).mean().item()

        return halfs, lasts

    def get_loss(self, pred, label, batch):
        loss = self.loss_func(pred - label)  # nnodes x layers
        # mean over each variable in an instance, then mean over instances
        loss = scatter(loss, batch, dim=0, reduce='mean').mean()
        return loss

    @classmethod
    def get_cos_sim(cls, pred, label, batch):
        # cosine similarity, only on the last layer
        pred_batch, _ = to_dense_batch(pred, batch)  # batchsize x max_nnodes x steps
        label_batch, _ = to_dense_batch(label, batch)  # batchsize x max_nnodes x steps
        target = pred_batch.new_ones(pred_batch.shape[0])
        cos = torch.vmap(cos_metric, in_dims=(2, 2, None), out_dims=1)(pred_batch, label_batch, target)  # batchsize x steps
        return cos

    @classmethod
    def violate_per_batch(cls, pred, data) -> torch.Tensor:
        Ax_minus_b = spmm(data['cons', 'to', 'vals'].edge_index,
                          data['cons', 'to', 'vals'].edge_attr.squeeze(),
                          data['cons'].num_nodes, data['vals'].num_nodes, pred).squeeze() - data.b
        violation = scatter(torch.abs(Ax_minus_b), data['cons'].batch, dim=0, reduce='mean')  # (batchsize,)
        return violation


class PlainGNNTrainer(Trainer):
    def __init__(self, loss_type, coeff_obj, coeff_vio):
        super().__init__(loss_type, 1., 1., 0.)
        self.coeff_obj = coeff_obj
        self.coeff_vio = coeff_vio

    def train(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        train_losses = 0.
        train_vios = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)

            pred, label = model(data)  # nnodes x steps
            loss = self.get_loss(pred, label, data['vals'].batch)
            loss_vio = (self.violate_per_batch(pred, data) ** 2).mean()

            train_losses += loss.detach() * data.num_graphs
            train_vios += loss_vio.detach() * data.num_graphs
            num_graphs += data.num_graphs

            if self.coeff_vio > 0:
                loss = loss * self.coeff_obj + loss_vio * self.coeff_vio

            # use both L2 loss and Cos similarity loss
            loss = loss / self.microbatch
            loss.backward()
            if (i + 1) % self.microbatch == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()

        return train_losses.item() / num_graphs, train_vios.item() / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        objgaps = []
        violations = []
        for i, data in enumerate(dataloader):
            data = data.to(device)
            _, obj_gap, violation = model.evaluation(data)

            objgaps.append(obj_gap)
            violations.append(violation)

        objgaps = torch.cat(objgaps, dim=0).mean().item()
        violations = torch.cat(violations, dim=0).mean().item()
        return objgaps, violations


class MultiGPUTrainer(Trainer):
    def __init__(self,
                 loss_type,
                 microbatch,
                 coeff_l2,
                 coeff_cos,
                 ):
        super().__init__(loss_type, microbatch, coeff_l2, coeff_cos)

    def train(self, dataloader, model, optimizer, local_device):
        model.train()
        optimizer.zero_grad()

        train_losses = 0.
        cos_sims = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(local_device)

            pred, label = model(data)  # nnodes x steps
            loss = self.get_loss(pred, label, data['vals'].batch)
            cos_sim = self.get_cos_sim(pred, label, data['vals'].batch)  # batchsize x steps

            train_losses += loss.detach() * data.num_graphs
            cos_sims = cos_sims + cos_sim.detach().sum(0)
            num_graphs += data.num_graphs

            # use both L2 loss and Cos similarity loss
            loss = self.coeff_l2 * loss + self.coeff_cos * cos_sim.mean()
            loss = loss / self.microbatch
            loss.backward()
            if (i + 1) % self.microbatch == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()

        return train_losses / num_graphs, cos_sims / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model, local_device):
        model.eval()
        lasts = []
        for i, data in enumerate(dataloader):
            data = data.to(local_device)
            _, best_obj_gap, obj_gaps, _, _ = model.evaluation(data)
            _, steps = obj_gaps.shape
            last_objgap = obj_gaps[:, -1]
            lasts.append(last_objgap)

        lasts = torch.cat(lasts, dim=0).mean()
        return lasts
