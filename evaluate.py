import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from ml_collections import ConfigDict
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm

from data.dataset import LPDataset
from utils.benchmark import sync_timer, gaussian_filter_bt
from data.collate_func import collate_fn_lp_bi
from utils.parsers import args_set_bool
from models.cycle_model import CycleGNN
from models.hetero_gnn import BipartiteHeteroGNN
from solver.customized_solver import ipm_overleaf
from solver.linprog import linprog


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--modelpath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', type=str, default='false')

    # model related
    parser.add_argument('--ipm_eval_steps', type=int, default=32)
    parser.add_argument('--conv', type=str, default='gcnconv')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=6)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='mlp layers within GENConv')
    parser.add_argument('--norm', type=str, default='graphnorm')  # empirically better

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    args = args_set_bool(vars(args))
    args = ConfigDict(args)

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath)[-10:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collate_fn = partial(collate_fn_lp_bi, device=device)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn)

    gnn_objgaps = []
    gnn_timsteps = []
    gnn_arange = []

    # warmup and set dimensions
    gnn = BipartiteHeteroGNN(conv=args.conv,
                             hid_dim=args.hidden,
                             num_conv_layers=args.num_conv_layers,
                             num_pred_layers=args.num_pred_layers,
                             num_mlp_layers=args.num_mlp_layers,
                             norm=args.norm)
    model = CycleGNN(1, args.ipm_eval_steps, gnn).to(device)
    data = next(iter(dataloader)).to(device)
    _ = gnn(data)

    for ckpt in [n for n in os.listdir(args.modelpath) if n.endswith('.pt')]:
        model.load_state_dict(torch.load(os.path.join(args.modelpath, ckpt), map_location=device))
        model.eval()

        for data in tqdm(dataloader):
            data = data.to(device)
            _, obj_gaps, time_stamps = model.evaluation(data, True)
            gnn_timsteps.append(time_stamps)
            gnn_objgaps.append(obj_gaps)
            gnn_arange.append(np.arange(1, args.ipm_eval_steps + 1))

    solver_objgaps = []
    solver_timsteps = []
    solver_steps = []
    solver_arange = []
    sp_objgaps = []
    sp_timsteps = []
    sp_steps = []
    sp_arange = []

    for data in tqdm(dataset):
        c = data.c.numpy()
        b = data.b.numpy()
        A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
                         col=data['cons', 'to', 'vals'].edge_index[1],
                         value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
                         is_sorted=True, trust_data=True).to_dense().numpy()
        opt_obj = data.obj_solution.item()

        start_t = sync_timer()
        res = ipm_overleaf(c, A, b, init='dumb')
        end_t = sync_timer()
        xs = np.stack(res['intermediate'], axis=0).dot(c)
        time_steps = np.arange(1, xs.shape[0] + 1) * (end_t - start_t) / xs.shape[0]
        solver_timsteps.append(time_steps)
        solver_objgaps.append(np.abs((xs - opt_obj) / (opt_obj + 1.e-6)))
        solver_steps.append(res['nit'])
        solver_arange.append(np.arange(1, xs.shape[0] + 1))

        start_t = sync_timer()
        res = linprog(
            c,
            A_ub=None, b_ub=None,
            A_eq=A, b_eq=b,
            bounds=None, method='interior-point', callback=lambda res: res.x)
        end_t = sync_timer()
        xs = np.stack(res.intermediate, axis=0).dot(c)
        time_steps = np.arange(1, xs.shape[0] + 1) * (end_t - start_t) / xs.shape[0]
        sp_timsteps.append(time_steps)
        sp_objgaps.append(np.abs((xs - opt_obj) / (opt_obj + 1.e-6)))
        sp_steps.append(res.nit)
        sp_arange.append(np.arange(1, xs.shape[0] + 1))

    gnn_timsteps = np.concatenate(gnn_timsteps, axis=0)
    gnn_objgaps = np.concatenate(gnn_objgaps, axis=0)
    gnn_arange = np.concatenate(gnn_arange, axis=0)
    solver_timsteps = np.concatenate(solver_timsteps, axis=0)
    solver_objgaps = np.concatenate(solver_objgaps, axis=0)
    solver_arange = np.concatenate(solver_arange, axis=0)
    sp_timsteps = np.concatenate(sp_timsteps, axis=0)
    sp_objgaps = np.concatenate(sp_objgaps, axis=0)
    sp_arange = np.concatenate(sp_arange, axis=0)

    # subplots
    fig, axes = plt.subplots(1, 2, figsize=(10,5), sharey=False)

    sort_idx = np.argsort(gnn_arange)
    gnn_arange = gnn_arange[sort_idx]
    gnn_objgaps_st = gnn_objgaps[sort_idx]
    g_st_mean, (g_st_low, g_st_upp) = gaussian_filter_bt(
        np.linspace(0, gnn_arange.max(), gnn_arange.shape[0]),
        gnn_arange, gnn_objgaps_st,
        1., n_boot=100)
    sns.lineplot(ax=axes[0], x=gnn_arange, y=g_st_mean, color='r', label='GNN')

    sort_idx = np.argsort(solver_arange)
    solver_arange = solver_arange[sort_idx]
    solver_objgaps_st = solver_objgaps[sort_idx]
    solver_st_mean, (solver_st_low, solver_st_upp) = gaussian_filter_bt(
        np.linspace(0, solver_arange.max(), solver_arange.shape[0]),
        solver_arange, solver_objgaps_st,
        1., n_boot=100)
    sns.lineplot(ax=axes[0], x=solver_arange, y=solver_st_mean, color='b', label='solver')

    sort_idx = np.argsort(sp_arange)
    sp_arange = sp_arange[sort_idx]
    sp_objgaps_st = sp_objgaps[sort_idx]
    sp_st_mean, (sp_st_low, sp_st_upp) = gaussian_filter_bt(
        np.linspace(0, sp_arange.max(), sp_arange.shape[0]),
        sp_arange, sp_objgaps_st,
        1., n_boot=100)
    sns.lineplot(ax=axes[0], x=sp_arange, y=sp_st_mean, color='g', label='scipy')

    sort_idx = np.argsort(gnn_timsteps)
    gnn_timsteps = gnn_timsteps[sort_idx]
    gnn_objgaps = gnn_objgaps[sort_idx]
    gnn_mean, (gnn_low, gnn_upp) = gaussian_filter_bt(np.linspace(0, gnn_timsteps.max(), gnn_timsteps.shape[0]),
                                                      gnn_timsteps, gnn_objgaps,
                                                      gnn_timsteps.max() / args.ipm_eval_steps, n_boot=10)

    sort_idx = np.argsort(solver_timsteps)
    solver_timsteps = solver_timsteps[sort_idx]
    solver_objgaps = solver_objgaps[sort_idx]
    solver_mean, (solver_low, solver_upp) = gaussian_filter_bt(np.linspace(0, solver_timsteps.max(), solver_timsteps.shape[0]),
                                                               solver_timsteps, solver_objgaps,
                                                               solver_timsteps.max() / np.mean(solver_steps), n_boot=10)

    sort_idx = np.argsort(sp_timsteps)
    sp_timsteps = sp_timsteps[sort_idx]
    sp_objgaps = sp_objgaps[sort_idx]
    sp_mean, (sp_low, sp_upp) = gaussian_filter_bt(np.linspace(0, sp_timsteps.max(), sp_timsteps.shape[0]),
                                                   sp_timsteps, sp_objgaps,
                                                   sp_timsteps.max() / np.mean(sp_steps), n_boot=10)

    sns.lineplot(ax=axes[1], x=gnn_timsteps, y=gnn_mean, label='GNN', color='r')
    axes[1].fill_between(gnn_timsteps, gnn_low, gnn_upp, color='r', alpha=0.5)
    sns.lineplot(ax=axes[1], x=solver_timsteps, y=solver_mean, label='solver', color='b')
    axes[1].fill_between(solver_timsteps, solver_low, solver_upp, color='b', alpha=0.5)
    sns.lineplot(ax=axes[1], x=sp_timsteps, y=sp_mean, label='scipy', color='g')
    axes[1].fill_between(sp_timsteps, sp_low, sp_upp, color='g', alpha=0.5)
    axes[0].set(yscale='log', ylim=[1.e-5, 1.], xlabel='Step', ylabel="Obj. rel error")
    axes[1].set(xscale='log', yscale='log', ylim=[1.e-5, 1.], xlabel='Time (sec)', ylabel="Obj. rel error")
    plt.legend()
    plt.savefig('temp.png', dpi=300)

    if args.use_wandb:
        wandb.log({"plot": wandb.Image(fig)})
