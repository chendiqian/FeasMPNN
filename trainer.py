import torch
import numpy as np
from torch_geometric.utils import to_dense_batch, scatter
from torch_sparse import spmm

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
    def eval(self, data_batch, model):
        model.eval()
        data_batch = data_batch.to(device)
        _, best_obj_gap, obj_gaps, _, _ = model.evaluation(data_batch)
        _, steps = obj_gaps.shape
        quarter_objgap = obj_gaps[:, steps // 4].mean()
        half_objgap = obj_gaps[:, steps // 2].mean()
        last_objgap = obj_gaps[:, -1].mean()
        return quarter_objgap, half_objgap, last_objgap

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

    @torch.no_grad()
    def eval_cons_violate(self, dataloader, model):
        model.eval()

        violations = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)
            pred, _ = model(data)
            violation = self.violate_per_batch(pred, data).sum()
            violations += violation
            num_graphs += data.num_graphs

        return violations / num_graphs

    @classmethod
    def violate_per_batch(cls, pred, data) -> torch.Tensor:
        Ax_minus_b = spmm(data['cons', 'to', 'vals'].edge_index,
                          data['cons', 'to', 'vals'].edge_attr.squeeze(),
                          data['cons'].num_nodes, data['vals'].num_nodes, pred).squeeze() - data.b
        violation = scatter(torch.abs(Ax_minus_b), data['cons'].batch, dim=0, reduce='mean')  # (batchsize,)
        return violation.cpu().numpy()


class IPMTrainer:
    def __init__(self,
                 loss_type,
                 ipm_steps,
                 ipm_alpha,
                 loss_weight):
        assert 0. <= ipm_alpha <= 1.
        self.ipm_steps = ipm_steps
        self.step_weight = torch.tensor([ipm_alpha ** (ipm_steps - l - 1)
                                         for l in range(ipm_steps)],
                                        dtype=torch.float, device=device)[None]
        # self.best_val_loss = 1.e8
        self.best_val_objgap = 100.
        self.best_val_consgap = 100.
        self.patience = 0
        self.loss_weight = loss_weight
        if loss_type == 'l2':
            self.loss_func = lambda x: x ** 2
        elif loss_type == 'l1':
            self.loss_func = lambda x: x.abs()
        else:
            raise ValueError

    def train(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            preds = model(data)
            loss = self.get_loss(preds, data)

            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=1.0,
                                           error_if_nonfinite=True)
            optimizer.step()

        return train_losses.item() / num_graphs

    def get_loss(self, vals, data):
        loss = 0.

        # primal
        primal_loss = (self.loss_func(
            vals[:, -self.ipm_steps:] -
            data.gt_primals[:, -self.ipm_steps:]
        ) * self.step_weight).mean()
        loss = loss + primal_loss * self.loss_weight['primal']

        # objgap
        obj_loss = (self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False)) * self.step_weight).mean()
        loss = loss + obj_loss * self.loss_weight['objgap']

        # cons
        constraint_gap = self.get_constraint_violation(vals, data)
        cons_loss = (self.loss_func(constraint_gap) * self.step_weight).mean()
        loss = loss + cons_loss * self.loss_weight['constraint']
        return loss

    def get_constraint_violation(self, vals, data):
        """
        Ax - b

        :param vals:
        :param data:
        :return:
        """
        pred = vals[:, -self.ipm_steps:]
        edge_index = data['cons', 'to', 'vals'].edge_index
        Ax = scatter(pred[edge_index[1], :] * data['cons', 'to', 'vals'].edge_attr, edge_index[0], reduce='sum', dim=0)
        constraint_gap = Ax - data.b[:, None]
        constraint_gap = torch.abs(constraint_gap)
        return constraint_gap

    def get_obj_metric(self, data, pred, hard_non_negative=False):
        # if hard_non_negative, we need a relu to make x all non-negative
        # just for metric usage, not for training
        pred = pred[:, -self.ipm_steps:]
        if hard_non_negative:
            pred = torch.relu(pred)
        c_times_x = data.c[:, None] * pred
        obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
        x_gt = data.gt_primals[:, -self.ipm_steps:]
        c_times_xgt = data.c[:, None] * x_gt
        obj_gt = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
        return (obj_pred - obj_gt) / obj_gt

    @torch.no_grad()
    def eval_metrics(self, dataloader, model):
        """
        both obj and constraint gap

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(device)
            vals = model(data)
            cons_gap.append(np.abs(self.get_constraint_violation(vals, data).detach().cpu().numpy()))
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=True).detach().cpu().numpy()))

        obj_gap = np.concatenate(obj_gap, axis=0)
        cons_gap = np.concatenate(cons_gap, axis=0)
        return obj_gap, cons_gap


class PDLPTrainer:
    def __init__(self, loss_type):
        self.best_val_objgap = 100.
        self.best_val_consgap = 100.
        self.patience = 0

        if loss_type == 'l2':
            self.loss_func = lambda x: x ** 2
        elif loss_type == 'l1':
            self.loss_func = lambda x: x.abs()
        else:
            raise ValueError

    def train(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            primals, duals = model(data)
            loss = self.get_loss(primals, duals, data)

            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=1.0,
                                           error_if_nonfinite=True)
            optimizer.step()

        return train_losses.item() / num_graphs

    def get_loss(self, primals, duals, data):
        loss = (self.loss_func(primals - data.primal_solution).mean() +
                self.loss_func(duals - data.dual_solution).mean())
        return loss

    def get_constraint_violation(self, vals, data):
        """
        Ax - b

        :param vals:
        :param data:
        :return:
        """
        edge_index = data['cons', 'to', 'vals'].edge_index
        Ax = scatter(vals[edge_index[1]] * data['cons', 'to', 'vals'].edge_attr.squeeze(), edge_index[0], reduce='sum', dim=0)
        constraint_gap = Ax - data.b[:, None]
        constraint_gap = torch.abs(constraint_gap)
        return constraint_gap

    def get_obj_metric(self, data, pred, hard_non_negative=False):
        # if hard_non_negative, we need a relu to make x all non-negative
        # just for metric usage, not for training
        if hard_non_negative:
            pred = torch.relu(pred)
        c_times_x = data.c * pred
        obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
        obj_gt = data.obj_solution
        return (obj_pred - obj_gt) / obj_gt

    @torch.no_grad()
    def eval_metrics(self, dataloader, model):
        """
        both obj and constraint gap

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(device)
            vals, _ = model(data)
            cons_gap.append(np.abs(self.get_constraint_violation(vals, data).detach().cpu().numpy()))
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=True).detach().cpu().numpy()))

        obj_gap = np.concatenate(obj_gap, axis=0).mean()
        cons_gap = np.concatenate(cons_gap, axis=0).mean()
        return obj_gap, cons_gap
