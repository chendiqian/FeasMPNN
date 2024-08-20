import argparse
import os
from functools import partial

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainer import Trainer
from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_bi
from data.transforms import GCNNorm
from models.cycle_model import CycleGNN
from models.hetero_gnn import BipartiteHeteroGNN


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--modelpath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', default=False, action='store_true')
    parser.add_argument('--plot', default=False, action='store_true')

    # model related
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--ipm_eval_steps', type=int, default=64)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--tau_scale', type=float, default=0.8)
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

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath, transform=GCNNorm() if 'gcn' in args.conv else None)[-1000:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collate_fn = partial(collate_fn_lp_bi, device=device)
    dataloader = DataLoader(dataset,
                            batch_size=args.batchsize,
                            shuffle=False,
                            collate_fn=collate_fn)

    gnn_objgaps = []
    best_gnn_obj = []
    gnn_timsteps = []
    gnn_times = []
    gnn_violations = []

    # warmup and set dimensions
    gnn = BipartiteHeteroGNN(conv=args.conv,
                             head=args.heads,
                             concat=args.concat,
                             hid_dim=args.hidden,
                             num_conv_layers=args.num_conv_layers,
                             num_pred_layers=args.num_pred_layers,
                             num_mlp_layers=args.num_mlp_layers,
                             norm=args.norm)
    model = CycleGNN(1, args.ipm_eval_steps, gnn, args.tau, args.tau_scale).to(device)
    with torch.no_grad():
        data = next(iter(dataloader)).to(device)
    _ = gnn(data)

    data = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for ckpt in [n for n in os.listdir(args.modelpath) if n.endswith('.pt')]:
        model.load_state_dict(torch.load(os.path.join(args.modelpath, ckpt), map_location=device))
        model.eval()

        gaps = []
        vios = []
        for data in tqdm(dataloader):
            data = data.to(device)
            final_x, _, obj_gaps, time_stamps = model.evaluation(data, True)
            gnn_timsteps.append(time_stamps)
            gnn_times.append(time_stamps[-1])
            gnn_objgaps.append(obj_gaps)
            gaps.append(obj_gaps[-1])
            vios.append(Trainer.violate_per_batch(final_x[:, None], data))
        gaps = np.concatenate(gaps, axis=0)
        vios = np.concatenate(vios)
        best_gnn_obj.append(np.mean(gaps))
        gnn_violations.append(np.mean(vios))

    time_per_step_gnn = [i[-1] / args.ipm_eval_steps for i in gnn_timsteps]

    stat_dict = {"gnn_obj_mean": np.mean(best_gnn_obj),
                 "gnn_obj_std": np.std(best_gnn_obj),
                 "gnn_violation_mean": np.mean(gnn_violations),
                 "gnn_violation_std": np.std(gnn_violations),
                 "gnn_time_per_step_mean": np.mean(time_per_step_gnn),
                 "gnn_time_per_step_std": np.std(time_per_step_gnn),
                 "gnn_time_mean": np.mean(gnn_times),
                 "gnn_time_std": np.std(gnn_times)}

    wandb.log(stat_dict)
