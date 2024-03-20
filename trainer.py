import torch
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch


class Trainer:
    def __init__(self,
                 device,
                 loss_type,
                 micro_batch):
        self.best_val_loss = 1.e8
        self.best_cos_sim = 1.e8
        self.patience = 0
        self.device = device
        self.loss_type = loss_type
        self.cos_metric = torch.nn.CosineEmbeddingLoss(reduction='none')
        if loss_type == 'l2':
            self.loss_func = torch.nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.loss_func = torch.nn.L1Loss(reduction='none')
        elif loss_type == 'cos':
            pass
        else:
            raise ValueError
        self.micro_batch = micro_batch

    def train(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        update_count = 0
        micro_batch = int(min(self.micro_batch, len(dataloader)))
        loss_scaling_lst = [micro_batch] * (len(dataloader) // micro_batch) + [len(dataloader) % micro_batch]

        train_losses = 0.
        cos_sims = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            pred = model(data)
            loss, cos_sim = self.get_loss(pred, data.label, data['vals'].batch)

            train_losses += loss.detach() * data.num_graphs
            cos_sims += cos_sim * data.num_graphs
            num_graphs += data.num_graphs

            update_count += 1
            loss = loss / float(loss_scaling_lst[0])  # scale the loss
            loss.backward()

            if update_count >= micro_batch or i == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0,
                                               error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()
                update_count = 0
                loss_scaling_lst.pop(0)

        return train_losses.item() / num_graphs, cos_sims.item() / num_graphs


    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        val_losses = 0.
        cos_sims = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            pred = model(data)
            loss, cos_sim = self.get_loss(pred, data.label, data['vals'].batch)
            val_losses += loss * data.num_graphs
            cos_sims += cos_sim * data.num_graphs
            num_graphs += data.num_graphs

        return val_losses.item() / num_graphs, cos_sims.item() / num_graphs

    def get_loss(self, pred, label, batch):
        # cosine similarity
        pred_batch, _ = to_dense_batch(pred, batch)
        label_batch, _ = to_dense_batch(label, batch)
        cos = self.cos_metric(pred_batch, label_batch, pred_batch.new_ones(pred_batch.shape[0]))
        cos = cos.mean()

        if self.loss_type in ['l1', 'l2']:
            loss = self.loss_func(pred, label)
            # mean over each variable in an instance, then mean over instances
            loss = scatter(loss, batch, dim=0, reduce='mean').mean()
        else:
            loss = cos
        return loss, cos.detach()
