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

    gnn_times = []
    oneshot_best_gnn_obj = []
    oneshot_gnn_violations = []
    search_best_gnn_obj = []
    search_gnn_violations = []

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

        os_gaps = []
        os_vios = []
        search_gaps = []
        search_vios = []

        pbar = tqdm(dataloader)
        for data in pbar:
            data = data.to(device)
            # oneshot prediction
            oneshot_prediction, oneshot_obj_gap, *_ = model.evaluation(data)
            os_vios.append(Trainer.violate_per_batch(oneshot_prediction[:, None], data))
            oneshot_obj_gap = oneshot_obj_gap.cpu().numpy()
            os_gaps.append(oneshot_obj_gap)

            # do the search
            project_x_objgap, final_x, obj_gaps, time_total = model.cycle_eval(data, args.ipm_eval_steps)
            gnn_times.append(time_total)
            search_gaps.append(obj_gaps[-1])
            search_vios.append(Trainer.violate_per_batch(final_x[:, None], data))

            pbar.set_postfix({'os_gap': oneshot_obj_gap.mean(),
                              'proj_gap': project_x_objgap,
                              'search_gap': obj_gaps[-1]})

        search_best_gnn_obj.append(np.mean(np.concatenate(search_gaps)))
        search_gnn_violations.append(np.mean(np.concatenate(search_vios)))
        oneshot_best_gnn_obj.append(np.mean(np.concatenate(os_gaps)))
        oneshot_gnn_violations.append(np.mean(np.concatenate(os_vios)))

    stat_dict = {"oneshot_obj_mean": np.mean(oneshot_best_gnn_obj),
                 "oneshot_obj_std": np.std(oneshot_best_gnn_obj),
                 "oneshot_vio_mean": np.mean(oneshot_gnn_violations),
                 "oneshot_vio_std": np.std(oneshot_gnn_violations),
                 "search_obj_mean": np.mean(search_best_gnn_obj),
                 "search_obj_std": np.std(search_best_gnn_obj),
                 "search_vio_mean": np.mean(search_gnn_violations),
                 "search_vio_std": np.std(search_gnn_violations),
                 "gnn_time_mean": np.mean(gnn_times),
                 "gnn_time_std": np.std(gnn_times)}

    wandb.log(stat_dict)
