import os
import argparse
from functools import partial

import copy
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_bi
from data.transforms import GCNNorm
from data.prefetch_generator import BackgroundGenerator
from models.hetero_gnn import BipartiteHeteroGNN
from models.cycle_model import CycleGNN
from trainer import Trainer
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
    parser.add_argument('--coeff_l2', type=float, default=0.1, help='balance between L2loss and cos loss')
    parser.add_argument('--coeff_cos', type=float, default=1., help='balance between L2loss and cos loss')

    # model related
    parser.add_argument('--ipm_train_steps', type=int, default=8)
    parser.add_argument('--ipm_eval_steps', type=int, default=64)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--tau_scale', type=float, default=0.5)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--conv', type=str, default='gcnconv')
    parser.add_argument('--heads', type=int, default=1, help='for GAT only')
    parser.add_argument('--concat', default=False, action='store_true', help='for GAT only')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=6)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='mlp layers within GENConv')
    parser.add_argument('--norm', type=str, default='graphnorm')  # empirically better

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath, transform=GCNNorm() if 'gcn' in args.conv else None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collate_fn = partial(collate_fn_lp_bi, device=device)
    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)],
                              batch_size=args.batchsize,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                            batch_size=1000,
                            shuffle=False,
                            collate_fn=collate_fn)
    val_batch = next(iter(val_loader))
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):],
                             batch_size=1000,
                             shuffle=False,
                             collate_fn=collate_fn)
    test_batch = next(iter(test_loader))

    # best_val_losses = []
    # best_val_cos_sims = []
    best_val_objgaps = []
    # test_losses = []
    # test_cos_sims = []
    test_objgaps = []

    for run in range(args.runs):
        gnn = BipartiteHeteroGNN(conv=args.conv,
                                 head=args.heads,
                                 concat=args.concat,
                                 hid_dim=args.hidden,
                                 num_conv_layers=args.num_conv_layers,
                                 num_pred_layers=args.num_pred_layers,
                                 num_mlp_layers=args.num_mlp_layers,
                                 norm=args.norm)
        model = CycleGNN(args.ipm_train_steps, args.ipm_eval_steps, gnn, args.tau, args.tau_scale).to(device)
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=70 // args.eval_every,
                                                         min_lr=1.e-5)

        trainer = Trainer(args.losstype, args.coeff_l2, args.coeff_cos)

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            (train_loss,
             train_cos_sim_first,
             train_cos_sim_last) = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)
            stats_dict = {'train_loss': train_loss,
                          'train_cos_sim_first': train_cos_sim_first,
                          "train_cos_sim_last": train_cos_sim_last,
                          'lr': scheduler.optimizer.param_groups[0]["lr"]}
            if epoch % args.eval_every == 0:
                val_obj_gap = trainer.eval(val_batch, model)
                if scheduler is not None:
                    scheduler.step(val_obj_gap)

                if trainer.best_objgap > val_obj_gap:
                    trainer.patience = 0
                    # trainer.best_val_loss = val_loss
                    # trainer.best_cos_sim = val_cos_sim
                    trainer.best_objgap = val_obj_gap
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt:
                        torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
                else:
                    trainer.patience += 1

                if trainer.patience > (args.patience // args.eval_every + 1):
                    break

                # stats_dict['val_loss'] = val_loss
                # stats_dict['val_cos_sim'] = val_cos_sim
                stats_dict['val_obj_gap'] = val_obj_gap

            pbar.set_postfix(stats_dict)
            wandb.log(stats_dict)
        # best_val_losses.append(trainer.best_val_loss)
        # best_val_cos_sims.append(trainer.best_cos_sim)
        best_val_objgaps.append(trainer.best_objgap)

        model.load_state_dict(best_model)
        test_obj_gap = trainer.eval(test_batch, model)
        # test_losses.append(test_loss)
        # test_cos_sims.append(test_cos_sim)
        test_objgaps.append(test_obj_gap)
        wandb.log({
            # 'test_loss': test_loss,
            # 'test_cos_sim': test_cos_sim,
            'test_obj_gap': test_obj_gap
        })

    wandb.log({
        # 'best_val_loss': np.mean(best_val_losses),
        # 'best_val_cos_sim': np.mean(best_val_cos_sims),
        'best_val_obj_gap': np.mean(best_val_objgaps),
        # 'test_loss_mean': np.mean(test_losses),
        # 'test_loss_std': np.std(test_losses),
        # 'test_cos_sim_mean': np.mean(test_cos_sims),
        # 'test_cos_sim_std': np.std(test_cos_sims),
        'test_obj_gap_mean': np.mean(test_objgaps),
        'test_obj_gap_std': np.std(test_objgaps),
    })
