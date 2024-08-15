import argparse
import os
from functools import partial

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainer import Trainer
from data.transforms import GCNNorm
from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_bi
from models.plain_gnn import BaseBipartiteHeteroGNN


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--modelpath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', default=False, action='store_true')

    # model related
    parser.add_argument('--ipm_eval_steps', type=int, default=64)
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
    # todo: add plain gnn eval

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath, transform=GCNNorm() if 'gcn' in args.conv else None)[-1000:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collate_fn = partial(collate_fn_lp_bi, device=device)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn)

    gnn_objgaps = []
    best_gnn_obj = []
    gnn_times = []
    gnn_violations = []

    # warmup and set dimensions
    model = BaseBipartiteHeteroGNN(conv=args.conv,
                                   head=args.heads,
                                   concat=args.concat,
                                   hid_dim=args.hidden,
                                   num_conv_layers=args.num_conv_layers,
                                   num_pred_layers=args.num_pred_layers,
                                   num_mlp_layers=args.num_mlp_layers,
                                   norm=args.norm).to(device)
    data = next(iter(dataloader)).to(device)
    _ = model(data)

    for ckpt in [n for n in os.listdir(args.modelpath) if n.endswith('.pt')]:
        model.load_state_dict(torch.load(os.path.join(args.modelpath, ckpt), map_location=device))
        model.eval()

        gaps = []
        vios = []
        for data in tqdm(dataloader):
            data = data.to(device)
            final_x, _, obj_gaps, time_total = model.cycle_eval(data, args.ipm_eval_steps)
            gnn_times.append(time_total)
            gnn_objgaps.append(obj_gaps)
            gaps.append(obj_gaps[-1])
            vios.append(Trainer.violate_per_batch(final_x.t(), data)[0].item())
        best_gnn_obj.append(np.mean(gaps))
        gnn_violations.append(np.mean(vios))

    gnn_violations = np.array(gnn_violations, dtype=np.float32)

    stat_dict = {"gnn_obj_mean": np.mean(best_gnn_obj),
                 "gnn_obj_std": np.std(best_gnn_obj),
                 "gnn_violation_mean": np.mean(gnn_violations),
                 "gnn_violation_std": np.std(gnn_violations),
                 "gnn_time_mean": np.mean(gnn_times),
                 "gnn_time_std": np.std(gnn_times)}

    if args.use_wandb:
        wandb.log(stat_dict)
