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
from data.collate_func import collate_fn_lp_base
from data.transforms import GCNNorm
from data.prefetch_generator import BackgroundGenerator
from trainer import IPMTrainer
from data.utils import save_run_config
from models.ipmgnn import BipartiteIPMGNN


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
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--conv', type=str, default='gcnconv')
    parser.add_argument('--heads', type=int, default=1, help='for GAT only')
    parser.add_argument('--concat', default=False, action='store_true', help='for GAT only')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=8)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--hid_pred', type=int, default=-1)
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='mlp layers within GENConv')
    parser.add_argument('--norm', type=str, default='graphnorm')  # empirically better

    # ipm specific
    parser.add_argument('--ipm_alpha', type=float, default=0.9)
    parser.add_argument('--ipm_steps', type=int, default=8)
    parser.add_argument('--loss_weight_x', type=float, default=1.0)
    parser.add_argument('--loss_weight_obj', type=float, default=1.0)
    parser.add_argument('--loss_weight_cons', type=float, default=1.0)  # does not work
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
    collate_fn = partial(collate_fn_lp_base)
    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)],
                              batch_size=args.batchsize,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                            batch_size=1000,
                            shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):],
                             batch_size=1000,
                             shuffle=False,
                             collate_fn=collate_fn)

    best_val_objgaps = []
    best_val_violations = []
    test_objgaps = []
    test_violations = []

    for run in range(args.runs):
        model = BipartiteIPMGNN(conv=args.conv,
                                head=args.heads,
                                concat=args.concat,
                                hid_dim=args.hidden,
                                num_conv_layers=args.num_conv_layers,
                                num_pred_layers=args.num_pred_layers,
                                hid_pred=args.hid_pred,
                                num_mlp_layers=args.num_mlp_layers,
                                norm=args.norm).to(device)
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=50 // args.eval_every,
                                                         min_lr=1.e-5)

        trainer = IPMTrainer(args.losstype,
                             args.ipm_steps,
                             args.ipm_alpha,
                             loss_weight={'primal': args.loss_weight_x,
                                          'objgap': args.loss_weight_obj,
                                          'constraint': args.loss_weight_cons})

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)

            with torch.no_grad():
                val_gaps, val_constraint_gap = trainer.eval_metrics(val_loader, model)

                # metric to cache the best model
                cur_mean_gap = val_gaps[:, -1].mean().item()
                cur_cons_gap_mean = val_constraint_gap[:, -1].mean().item()
                if scheduler is not None:
                    scheduler.step(cur_mean_gap)

                if trainer.best_val_objgap > cur_mean_gap:
                    trainer.patience = 0
                    trainer.best_val_objgap = cur_mean_gap
                    trainer.best_val_consgap = cur_cons_gap_mean
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt:
                        torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
                else:
                    trainer.patience += 1

            if trainer.patience > (args.patience // args.eval_every + 1):
                break

            pbar.set_postfix({'train_loss': train_loss,
                              'val_obj': cur_mean_gap,
                              'val_cons': cur_cons_gap_mean,
                              'lr': scheduler.optimizer.param_groups[0]["lr"]})
            log_dict = {'train_loss': train_loss,
                        'val_obj_gap_last_mean': cur_mean_gap,
                        'val_cons_gap_last_mean': cur_cons_gap_mean,
                        'lr': scheduler.optimizer.param_groups[0]["lr"]}

            wandb.log(log_dict)
        # best_val_losses.append(trainer.best_val_loss)
        best_val_objgaps.append(trainer.best_val_objgap)
        best_val_violations.append(trainer.best_val_consgap)

        model.load_state_dict(best_model)
        with torch.no_grad():
            test_gaps, test_cons_gap = trainer.eval_metrics(test_loader, model)
        test_objgaps.append(test_gaps[:, -1].mean().item())
        test_violations.append(test_cons_gap[:, -1].mean().item())

        wandb.log({'test_objgap': test_objgaps[-1]})
        wandb.log({'test_consgap': test_violations[-1]})

    wandb.log({
        'best_val_objgap': np.mean(best_val_objgaps),
        'test_objgap_mean': np.mean(test_objgaps),
        'test_objgap_std': np.std(test_objgaps),
        'test_consgap_mean': np.mean(test_violations),
        'test_consgap_std': np.std(test_violations),
        'test_hybrid_gap': np.mean(test_objgaps) + np.mean(test_consgap_mean),  # for the sweep
    })
