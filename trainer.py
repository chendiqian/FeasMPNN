import torch
from torch_scatter import scatter


class Trainer:
    def __init__(self,
                 device,
                 loss_type,
                 micro_batch):
        self.best_val_loss = 1.e8
        self.patience = 0
        self.device = device
        self.loss_type = loss_type
        if loss_type == 'l2':
            self.loss_func = lambda x: x ** 2
        elif loss_type == 'l1':
            self.loss_func = lambda x: x.abs()
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
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            pred = model(data)  # nnodes x steps
            loss = self.get_loss(pred, data['vals'].batch)

            train_losses += loss.detach() * data.num_graphs
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

        return train_losses.item() / num_graphs


    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        val_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            pred = model(data)
            loss = self.get_loss(pred, data['vals'].batch)
            val_losses += loss * data.num_graphs
            num_graphs += data.num_graphs

        return val_losses.item() / num_graphs

    def get_loss(self, pred, batch):
        loss = self.loss_func(pred)  # nnodes x layers
        # mean over each variable in an instance, then mean over instances
        loss = scatter(loss, batch, dim=0, reduce='mean').mean()
        return loss
