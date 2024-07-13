import argparse
import os
from functools import partial

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm

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
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=6)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='mlp layers within GENConv')
    parser.add_argument('--norm', type=str, default='graphnorm')  # empirically better

    return parser.parse_args()


if __name__ == '__main__':
    raise ValueError("makes no sense right now, cause plain GNN does not encode starting position!")
    args = args_parser()

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath)[:10]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collate_fn = partial(collate_fn_lp_bi, device=device)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn)

    gnn_objgaps = []
    gnn_timsteps = []
    gnn_violation = []

    # warmup and set dimensions
    model = BaseBipartiteHeteroGNN(conv=args.conv,
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

        for data in tqdm(dataloader):
            data = data.to(device)
            final_x, _, obj_gaps, time_stamps = model.cycle_eval(data, args.ipm_eval_steps)
            final_x = final_x.squeeze(0)  # 1 graph, so 1 x nnodes
            gnn_timsteps.append(time_stamps)
            gnn_objgaps.append(obj_gaps)

            c = data.c.numpy()
            b = data.b.numpy()
            A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
                             col=data['cons', 'to', 'vals'].edge_index[1],
                             value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
                             is_sorted=True, trust_data=True).to_dense().cpu().numpy()
            Ax_minus_b = (A * final_x.cpu().numpy()[None]).sum(1) - b
            gnn_violation.append(np.abs(Ax_minus_b).mean())

    best_gnn_obj = [i[-1] for i in gnn_objgaps]
    time_per_step_gnn = [i[-1] / args.ipm_eval_steps for i in gnn_timsteps]

    gnn_timsteps = np.concatenate(gnn_timsteps, axis=0)
    gnn_objgaps = np.concatenate(gnn_objgaps, axis=0)
    gnn_violation = np.array(gnn_violation, dtype=np.float32)

    stat_dict = {"gnn_obj_mean": np.mean(best_gnn_obj),
                 "gnn_obj_std": np.std(best_gnn_obj),
                 "gnn_time_per_step_mean": np.mean(time_per_step_gnn),
                 "gnn_time_per_step_std": np.std(time_per_step_gnn),
                 "gnn_violation_mean": np.mean(gnn_violation),
                 "gnn_violation_std": np.std(gnn_violation),
                 }

    if args.use_wandb:
        wandb.log(stat_dict)
