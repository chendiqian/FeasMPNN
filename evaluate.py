import os
import argparse

from ml_collections import ConfigDict
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.dataset import LPDataset
from data.utils import args_set_bool, collate_fn_lp, gaussian_filter_bt
from models.hetero_gnn import TripartiteHeteroGNN
from models.cycle_model import CycleGNN


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--modelpath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', type=str, default='false')

    # model related
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

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath)[-10:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=partial(collate_fn_lp, device=device))

    gnn_objgaps = []
    gnn_timsteps = []

    # warmup and set dimensions
    gnn = TripartiteHeteroGNN(conv=args.conv,
                              hid_dim=args.hidden,
                              num_conv_layers=args.num_conv_layers,
                              num_pred_layers=args.num_pred_layers,
                              num_mlp_layers=args.num_mlp_layers,
                              dropout=args.dropout,
                              norm=args.norm,
                              use_res=args.use_res,
                              conv_sequence=args.conv_sequence)
    model = CycleGNN(1, args.ipm_eval_steps, gnn).to(device)
    data = next(iter(dataloader))
    _ = gnn(data)

    for ckpt in [n for n in os.listdir(args.modelpath) if n.endswith('.pt')]:
        model.load_state_dict(torch.load(os.path.join(args.modelpath, ckpt), map_location=device))
        model.eval()

        for data in tqdm(dataloader):
            data = data.to(device)
            _, obj_gaps, time_stamps = model.evaluation(data, True)
            gnn_timsteps.append(time_stamps)
            gnn_objgaps.append(obj_gaps)

    gnn_timsteps = np.concatenate(gnn_timsteps, axis=0)
    gnn_objgaps = np.concatenate(gnn_objgaps, axis=0)
    sort_idx = np.argsort(gnn_timsteps)
    gnn_timsteps = gnn_timsteps[sort_idx]
    gnn_objgaps = gnn_objgaps[sort_idx]

    time_grid = np.linspace(0, gnn_timsteps.max(), gnn_timsteps.shape[0])
    sigma = gnn_timsteps.max() / args.ipm_eval_steps
    mean, (low, upp) = gaussian_filter_bt(time_grid, gnn_timsteps, gnn_objgaps, sigma, n_boot=10)

    ax = sns.lineplot(x=gnn_timsteps, y=mean)
    ax.fill_between(gnn_timsteps, low, upp, alpha=0.5)

    plt.savefig('temp.png', dpi=300)
