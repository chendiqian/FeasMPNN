import os
import argparse
import yaml
from functools import partial

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
from models.plain_gnn import BaseBipartiteHeteroGNN
from trainer import Trainer


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
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--val_batchsize', type=int, default=1024)
    parser.add_argument('--loss_lambda', type=float, default=1., help='balance between L2loss and cos loss')

    # model related
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--conv', type=str, default='gcnconv')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=6)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='mlp layers within GENConv')
    parser.add_argument('--norm', type=str, default='graphnorm')  # empirically better

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()

    if args.ckpt:
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        exist_runs = [d for d in os.listdir('logs') if d.startswith(args.wandbname)]
        log_folder_name = f'logs/{args.wandbname}exp{len(exist_runs)}'
        os.mkdir(log_folder_name)
        with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
            yaml.dump(vars(args), outfile, default_flow_style=False)

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath, transform=GCNNorm() if args.conv.startswith('gcn') else None)
    # remove unnecessary for training
    dataset._data.A_col = None
    dataset._data.A_row = None
    dataset._data.A_val = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collate_fn = partial(collate_fn_lp_base)
    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)],
                              batch_size=args.batchsize,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                            batch_size=args.val_batchsize,
                            shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):],
                             batch_size=args.val_batchsize,
                             shuffle=False,
                             collate_fn=collate_fn)

    best_val_losses = []
    best_val_cos_sims = []
    best_val_objgaps = []
    test_losses = []
    test_cos_sims = []
    test_objgaps = []
    test_violations = []

    for run in range(args.runs):
        if args.ckpt:
            os.mkdir(os.path.join(log_folder_name, f'run{run}'))
        model = BaseBipartiteHeteroGNN(conv=args.conv,
                                       hid_dim=args.hidden,
                                       num_conv_layers=args.num_conv_layers,
                                       num_pred_layers=args.num_pred_layers,
                                       num_mlp_layers=args.num_mlp_layers,
                                       norm=args.norm).to(device)
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=50 // args.eval_every,
                                                         min_lr=1.e-5)

        trainer = Trainer(args.losstype, args.loss_lambda)

        test_violation = trainer.eval_cons_violate(BackgroundGenerator(train_loader, device, 4), model)

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss, train_cos_sim = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)
            stats_dict = {'train_loss': train_loss,
                          'train_cos_sim': train_cos_sim,
                          'lr': scheduler.optimizer.param_groups[0]["lr"]}
            if epoch % args.eval_every == 0:
                val_loss, val_cos_sim, val_obj_gap = trainer.eval(BackgroundGenerator(val_loader, device, 4), model)

                if scheduler is not None:
                    scheduler.step(val_obj_gap)

                if trainer.best_objgap > val_obj_gap:
                    trainer.patience = 0
                    trainer.best_val_loss = val_loss
                    trainer.best_cos_sim = val_cos_sim
                    trainer.best_objgap = val_obj_gap
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt:
                        torch.save(model.state_dict(), os.path.join(log_folder_name, f'run{run}', 'best_model.pt'))
                else:
                    trainer.patience += 1

                if trainer.patience > (args.patience // args.eval_every + 1):
                    break

                stats_dict['val_loss'] = val_loss
                stats_dict['val_cos_sim'] = val_cos_sim
                stats_dict['val_obj_gap'] = val_obj_gap

            pbar.set_postfix(stats_dict)
            wandb.log(stats_dict)
        best_val_losses.append(trainer.best_val_loss)
        best_val_cos_sims.append(trainer.best_cos_sim)
        best_val_objgaps.append(trainer.best_objgap)

        model.load_state_dict(best_model)
        test_loss, test_cos_sim, test_obj_gap = trainer.eval(test_loader, model)
        test_violation = trainer.eval_cons_violate(BackgroundGenerator(train_loader, device, 4), model)
        test_losses.append(test_loss)
        test_cos_sims.append(test_cos_sim)
        test_objgaps.append(test_obj_gap)
        test_violations.append(test_violation)
        wandb.log({'test_loss': test_loss,
                   'test_cos_sim': test_cos_sim,
                   'test_obj_gap': test_obj_gap,
                   'test_violation': test_violation})

    wandb.log({
        'best_val_loss': np.mean(best_val_losses),
        'best_val_cos_sim': np.mean(best_val_cos_sims),
        'best_val_obj_gap': np.mean(best_val_objgaps),
        'test_loss_mean': np.mean(test_losses),
        'test_loss_std': np.std(test_losses),
        'test_cos_sim_mean': np.mean(test_cos_sims),
        'test_cos_sim_std': np.std(test_cos_sims),
        'test_obj_gap_mean': np.mean(test_objgaps),
        'test_obj_gap_std': np.std(test_objgaps),
        'test_violation_mean': np.mean(test_violations),
        'test_violation_std': np.std(test_violations),
    })
