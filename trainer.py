import torch
from torch_geometric.utils import to_dense_batch, scatter
from torch_sparse import spmm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cos_metric = torch.nn.CosineEmbeddingLoss(reduction='none')


class Trainer:
    def __init__(self,
                 loss_type,
                 coeff_l2,
                 coeff_cos,
                 return_tensor=False  # for multi gpu
                 ):
        self.best_val_loss = 1.e8
        self.best_cos_sim = 1.e8
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
        self.return_tensor = return_tensor

    def train(self, dataloader, model, optimizer, local_device = None):
        if local_device is None:
            local_device = device
        model.train()
        optimizer.zero_grad()

        train_losses = 0.
        cos_sims = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(local_device)
            optimizer.zero_grad()

            pred, label = model(data)  # nnodes x steps
            loss = self.get_loss(pred, label, data['vals'].batch)
            cos_sim = self.get_cos_sim(pred, label, data['vals'].batch)

            train_losses += loss.detach() * data.num_graphs
            cos_sims += cos_sim.detach() * data.num_graphs
            num_graphs += data.num_graphs

            # use both L2 loss and Cos similarity loss
            loss = self.coeff_l2 * loss + self.coeff_cos * cos_sim
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        train_losses = train_losses / num_graphs
        cos_sims = cos_sims / num_graphs
        if not self.return_tensor:
            train_losses = train_losses.item()
            cos_sims = cos_sims.item()
        return train_losses, cos_sims

    @torch.no_grad()
    def eval(self, dataloader, model, local_device = None):
        if local_device is None:
            local_device = device
        model.eval()

        val_losses = 0.
        cos_sims = 0.
        num_graphs = 0
        obj_gaps = []
        for i, data in enumerate(dataloader):
            data = data.to(local_device)
            pred, label = model(data)
            loss = self.get_loss(pred, label, data['vals'].batch)
            cos_sim = self.get_cos_sim(pred, label, data['vals'].batch)

            val_losses += loss * data.num_graphs
            cos_sims += cos_sim * data.num_graphs
            num_graphs += data.num_graphs

            _, obj_gap, _, _ = model.evaluation(data)
            obj_gaps.append(obj_gap)

        objs = torch.cat(obj_gaps, dim=0).mean()
        val_losses = val_losses / num_graphs
        cos_sims = cos_sims / num_graphs
        if not self.return_tensor:
            objs = objs.item()
            val_losses = val_losses.item()
            cos_sims = cos_sims.item()
        return val_losses, cos_sims, objs

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
        cos = torch.vmap(cos_metric, in_dims=(2, 2, None), out_dims=1)(pred_batch, label_batch, target)
        cos = cos.mean()
        return cos

    @torch.no_grad()
    def eval_cons_violate(self, dataloader, model, local_device = None):
        if local_device is None:
            local_device = device
        model.eval()

        violations = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(local_device)
            pred, _ = model(data)
            violation = self.violate_per_batch(pred, data).sum()
            violations += violation
            num_graphs += data.num_graphs

        return violations.item() / num_graphs

    @classmethod
    def violate_per_batch(cls, pred, data) -> torch.Tensor:
        Ax_minus_b = spmm(data['cons', 'to', 'vals'].edge_index,
                          data['cons', 'to', 'vals'].edge_attr.squeeze(),
                          data['cons'].num_nodes, data['vals'].num_nodes, pred).squeeze() - data.b
        violation = scatter(torch.abs(Ax_minus_b), data['cons'].batch, dim=0, reduce='mean')  # (batchsize,)
        return violation
