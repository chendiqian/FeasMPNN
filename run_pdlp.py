import os
import argparse

import copy
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_base
from data.transforms import GCNNorm
from data.prefetch_generator import BackgroundGenerator
from models.pdhg_net import PDHGNet
from trainer import PDLPTrainer
from data.utils import save_run_config


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', default=False, action='store_true')

    # training dynamics
    parser.add_argument('--losstype', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--ckpt', default=False, action='store_true')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=300)
    parser.add_argument('--batchsize', type=int, default=32)

    # model related
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--conv', type=str, default='gcnconv')
    parser.add_argument('--heads', type=int, default=1, help='for GAT only')
    parser.add_argument('--concat', default=False, action='store_true', help='for GAT only')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=6)

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath, transform=GCNNorm() if 'gcn' in args.conv else None)[:20]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)],
                              batch_size=args.batchsize,
                              shuffle=True,
                              collate_fn=collate_fn_lp_base)
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                            batch_size=1000,
                            shuffle=False,
                            collate_fn=collate_fn_lp_base)
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):],
                             batch_size=1000,
                             shuffle=False,
                             collate_fn=collate_fn_lp_base)

    best_val_objgaps = []
    test_objgaps = []

    for run in range(args.runs):
        model = PDHGNet(args.conv, args.hidden, args.num_conv_layers).to(device)
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=70 // args.eval_every,
                                                         min_lr=1.e-5)

        trainer = PDLPTrainer(args.losstype)

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)
            stats_dict = {'train_loss': train_loss,
                          'lr': scheduler.optimizer.param_groups[0]["lr"]}
            if epoch % args.eval_every == 0:
                val_obj_gap, val_constraint_gap = trainer.eval_metrics(val_loader, model)
                if scheduler is not None:
                    scheduler.step(val_obj_gap)

                if trainer.best_val_objgap > val_obj_gap:
                    trainer.best_val_objgap = val_obj_gap
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt:
                        torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
                else:
                    trainer.patience += 1

                if trainer.patience > (args.patience // args.eval_every + 1):
                    break

                stats_dict['val_obj_gap'] = val_obj_gap

            pbar.set_postfix(stats_dict)
            wandb.log(stats_dict)
        best_val_objgaps.append(trainer.best_val_objgap)

        model.load_state_dict(best_model)
        test_obj_gap = trainer.eval_metrics(test_loader, model)
        test_objgaps.append(test_obj_gap)
        wandb.log({
            'test_obj_gap': test_obj_gap
        })

    wandb.log({
        'best_val_obj_gap': np.mean(best_val_objgaps),
        'test_obj_gap_mean': np.mean(test_objgaps),
        'test_obj_gap_std': np.std(test_objgaps),
    })
