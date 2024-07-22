import torch
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter
from torch_sparse import spmm


class Trainer:
    def __init__(self,
                 loss_type,
                 loss_lambda):
        self.best_val_loss = 1.e8
        self.best_cos_sim = 1.e8
        self.best_objgap = 1.e8
        self.patience = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_type = loss_type
        self.cos_metric = torch.nn.CosineEmbeddingLoss(reduction='none')
        if loss_type == 'l2':
            self.loss_func = lambda x: x ** 2
        elif loss_type == 'l1':
            self.loss_func = lambda x: x.abs()
        elif loss_type == 'cos':
            pass
        else:
            raise ValueError
        self.loss_lambda = loss_lambda

    def train(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        train_losses = 0.
        cos_sims = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            pred, label = model(data)  # nnodes x steps
            loss = self.get_loss(pred, label, data['vals'].batch)
            cos_sim = self.get_cos_sim(pred, label, data['vals'].batch)

            train_losses += loss.detach() * data.num_graphs
            cos_sims += cos_sim.detach() * data.num_graphs
            num_graphs += data.num_graphs

            # use both L2 loss and Cos similarity loss
            loss = loss + self.loss_lambda * cos_sim
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=1.0,
                                           error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad()

        return train_losses.item() / num_graphs, cos_sims.item() / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        val_losses = 0.
        cos_sims = 0.
        num_graphs = 0
        obj_gaps = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            pred, label = model(data)
            loss = self.get_loss(pred, label, data['vals'].batch)
            cos_sim = self.get_cos_sim(pred, label, data['vals'].batch)

            val_losses += loss * data.num_graphs
            cos_sims += cos_sim * data.num_graphs
            num_graphs += data.num_graphs

            obj_gap, *_ = model.evaluation(data)
            obj_gaps.append(obj_gap)

        objs = torch.cat(obj_gaps, dim=0).mean().item()
        return val_losses.item() / num_graphs, cos_sims.item() / num_graphs, objs

    def get_loss(self, pred, label, batch):
        loss = self.loss_func(pred - label)  # nnodes x layers
        # mean over each variable in an instance, then mean over instances
        loss = scatter(loss, batch, dim=0, reduce='mean').mean()
        return loss

    def get_cos_sim(self, pred, label, batch):
        # cosine similarity, only on the last layer
        pred_batch, _ = to_dense_batch(pred, batch)  # batchsize x max_nnodes x steps
        label_batch, _ = to_dense_batch(label, batch)  # batchsize x max_nnodes x steps
        target = pred_batch.new_ones(pred_batch.shape[0])
        cos = torch.vmap(self.cos_metric, in_dims=(2, 2, None), out_dims=1)(pred_batch, label_batch, target)
        cos = cos.mean()
        return cos

    @torch.no_grad()
    def eval_cons_violate(self, dataloader, model):
        model.eval()

        violations = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            pred, _ = model(data)
            Ax_minus_b = spmm(data['cons', 'to', 'vals'].edge_index,
                              data['cons', 'to', 'vals'].edge_attr.squeeze(),
                              data['cons'].num_nodes, data['vals'].num_nodes, pred).squeeze() - data.b
            violation = scatter(torch.abs(Ax_minus_b), data['cons'].batch, dim=0, reduce='mean').sum()
            violations += violation
            num_graphs += data.num_graphs

        return violations.item() / num_graphs
