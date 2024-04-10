import os
import argparse
from ml_collections import ConfigDict
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
from data.utils import args_set_bool, collate_fn_lp
from data.prefetch_generator import BackgroundGenerator
from models.hetero_gnn import TripartiteHeteroGNN
from models.cycle_model import CycleGNN
from trainer import Trainer


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', type=str, default='false')

    # training dynamics
    parser.add_argument('--losstype', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--ckpt', type=str, default='true')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--micro_batch', type=int, default=1)

    # model related
    parser.add_argument('--ipm_train_steps', type=int, default=8)
    parser.add_argument('--ipm_eval_steps', type=int, default=64)
    parser.add_argument('--conv', type=str, default='gcnconv')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=6)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='mlp layers within GENConv')
    parser.add_argument('--conv_sequence', type=str, default='cov')
    parser.add_argument('--norm', type=str, default='graphnorm')  # empirically better
    parser.add_argument('--use_res', type=str, default='false')  # does not help
    parser.add_argument('--dropout', type=float, default=0.)  # must

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    args = args_set_bool(vars(args))
    args = ConfigDict(args)

    if args.ckpt:
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        exist_runs = [d for d in os.listdir('logs') if d.startswith('exp')]
        log_folder_name = f'logs/exp{len(exist_runs)}'
        os.mkdir(log_folder_name)
        with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
            yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath)
    # remove unnecessary for training
    dataset._data.A_col = None
    dataset._data.A_row = None
    dataset._data.A_val = None
    dataset._data.b = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)],
                              batch_size=args.batchsize,
                              shuffle=True,
                              collate_fn=partial(collate_fn_lp, device=device))
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                            batch_size=args.batchsize * 2,
                            shuffle=False,
                            collate_fn=partial(collate_fn_lp, device=device))
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):],
                             batch_size=args.batchsize * 2,
                             shuffle=False,
                             collate_fn=partial(collate_fn_lp, device=device))

    best_val_losses = []
    best_val_cos_sims = []
    best_val_objgaps = []
    test_losses = []
    test_cos_sims = []
    test_objgaps = []

    for run in range(args.runs):
        if args.ckpt:
            os.mkdir(os.path.join(log_folder_name, f'run{run}'))
        gnn = TripartiteHeteroGNN(conv=args.conv,
                                  hid_dim=args.hidden,
                                  num_conv_layers=args.num_conv_layers,
                                  num_pred_layers=args.num_pred_layers,
                                  num_mlp_layers=args.num_mlp_layers,
                                  dropout=args.dropout,
                                  norm=args.norm,
                                  use_res=args.use_res,
                                  conv_sequence=args.conv_sequence)
        model = CycleGNN(args.ipm_train_steps, args.ipm_eval_steps, gnn).to(device)
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1.e-5)

        trainer = Trainer(device, args.losstype, args.micro_batch)

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            # train_loss, train_cos_sim = trainer.train(train_loader, model, optimizer)
            # val_loss, val_cos_sim = trainer.eval(val_loader, model)
            # val_obj_gap = trainer.get_pseudo_ipm_solution(val_loader, model)

            train_loss, train_cos_sim = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)
            val_loss, val_cos_sim = trainer.eval(BackgroundGenerator(val_loader, device, 4), model)
            val_obj_gap = trainer.get_pseudo_ipm_solution(BackgroundGenerator(val_loader, device, 4), model)

            # imo cosine similarity makes more sense, we don't care about norm but direction
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

            if trainer.patience > args.patience:
                break

            pbar.set_postfix({'train_loss': train_loss,
                              'train_cos_sim': train_cos_sim,
                              'val_loss': val_loss,
                              'val_cos_sim': val_cos_sim,
                              'val_obj_gap': val_obj_gap,
                              'lr': scheduler.optimizer.param_groups[0]["lr"]})
            log_dict = {'train_loss': train_loss,
                        'train_cos_sim': train_cos_sim,
                        'val_loss': val_loss,
                        'val_cos_sim': val_cos_sim,
                        'val_obj_gap': val_obj_gap,
                        'lr': scheduler.optimizer.param_groups[0]["lr"]}
            wandb.log(log_dict)
        best_val_losses.append(trainer.best_val_loss)
        best_val_cos_sims.append(trainer.best_cos_sim)
        best_val_objgaps.append(trainer.best_objgap)

        model.load_state_dict(best_model)
        test_loss, test_cos_sim = trainer.eval(test_loader, model)
        test_obj_gap = trainer.get_pseudo_ipm_solution(test_loader, model)
        test_losses.append(test_loss)
        test_cos_sims.append(test_cos_sim)
        test_objgaps.append(test_obj_gap)
        wandb.log({'test_loss': test_loss, 'test_cos_sim': test_cos_sim, 'test_obj_gap': test_obj_gap})

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
    })
