import os

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.collate_func import collate_fn_lp_base
from data.dataset import LPDataset
from data.transforms import GCNNorm
from models.ipm_model import IPMGNN
from models.hetero_gnn import TripartiteHeteroGNN
from trainer import Trainer


@hydra.main(version_base=None, config_path='./config', config_name="evaluate_ipm")
def main(args: DictConfig):
    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath, transform=GCNNorm() if 'gcn' in args.conv else None)[-100:]
    if args.debug:
        dataset = dataset[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = DataLoader(dataset,
                            batch_size=args.batchsize,
                            shuffle=False,
                            collate_fn=collate_fn_lp_base)

    best_gnn_obj = []
    gnn_timsteps = []
    gnn_times = []
    gnn_violations = []

    # warmup and set dimensions
    gnn = TripartiteHeteroGNN(conv=args.conv,
                              head=args.gat.heads,
                              concat=args.gat.concat,
                              hid_dim=args.hidden,
                              num_encode_layers=args.num_encode_layers,
                              num_conv_layers=args.num_conv_layers,
                              num_pred_layers=args.num_pred_layers,
                              hid_pred=args.hid_pred,
                              num_mlp_layers=args.num_mlp_layers,
                              norm=args.norm,
                              plain_xstarts=args.plain_xstarts)
    model = IPMGNN(1, args.ipm_eval_steps, gnn).to(device)

    # warm up
    with torch.no_grad():
        data = next(iter(dataloader)).to(device)
        for _ in range(10):
            _ = gnn(data, data.x_feasible)

    # end warming up
    del data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # prep pretrained model
    if args.modelpath is not None:
        model_list = [n for n in os.listdir(args.modelpath) if n.endswith('.pt')]
    else:
        model_list = [None]

    # begin evaluation
    for ckpt in model_list:
        if ckpt is not None:
            model.load_state_dict(torch.load(os.path.join(args.modelpath, ckpt), map_location=device))
        model.eval()

        gaps = []
        vios = []
        times = []
        pbar = tqdm(dataloader)
        for data in pbar:
            data = data.to(device)
            final_x, best_obj, time_stamps = model.evaluation(data)
            gnn_timsteps.append(time_stamps)
            times.append(time_stamps[-1])
            best_obj = best_obj.cpu().numpy()
            gaps.append(best_obj)
            vios.append(Trainer.violate_per_batch(final_x[:, None], data).cpu().numpy())

            stat_dict = {'gap': best_obj.mean(),
                         'vio': vios[-1].mean(),
                         'gnn_time': time_stamps[-1]}
            pbar.set_postfix(stat_dict)
            wandb.log(stat_dict)
        gaps = np.concatenate(gaps, axis=0)
        vios = np.concatenate(vios)
        best_gnn_obj.append(np.mean(gaps))
        gnn_violations.append(np.mean(vios))
        gnn_times.append(np.mean(times))

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


if __name__ == '__main__':
    main()
