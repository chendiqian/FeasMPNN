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
from models.ipm_fixstep_model import FixStepBipartiteIPMGNN, FixStepTripartiteIPMGNN
from data.utils import sync_timer


@hydra.main(version_base=None, config_path='./config', config_name="evaluate_fixstep_ipm")
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
    gnn_times = []
    gnn_violations = []

    # warmup and set dimensions
    ModelClass = FixStepTripartiteIPMGNN if args.tripartite else FixStepBipartiteIPMGNN
    model = ModelClass(conv=args.conv,
                       head=args.gat.heads,
                       concat=args.gat.concat,
                       hid_dim=args.hidden,
                       num_encode_layers=args.num_encode_layers,
                       num_conv_layers=args.num_conv_layers,
                       num_pred_layers=args.num_pred_layers,
                       hid_pred=args.hid_pred,
                       num_mlp_layers=args.num_mlp_layers,
                       norm=args.norm).to(device)

    # warm up
    with torch.no_grad():
        data = next(iter(dataloader)).to(device)
        for _ in range(10):
            _ = model(data)

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
            t1 = sync_timer()
            final_x, best_obj, violations = model.evaluation(data)
            t2 = sync_timer()
            times.append(t2 - t1)
            best_obj = best_obj.cpu().numpy()
            violations = violations.cpu().numpy()
            gaps.append(best_obj)
            vios.append(violations)

            stat_dict = {'gap': best_obj.mean(),
                         'vio': violations.mean()}
            pbar.set_postfix(stat_dict)
            wandb.log(stat_dict)
        gaps = np.concatenate(gaps, axis=0)
        vios = np.concatenate(vios)
        best_gnn_obj.append(np.mean(gaps))
        gnn_violations.append(np.mean(vios))
        gnn_times.append(np.mean(times))

    stat_dict = {"gnn_obj_mean": np.mean(best_gnn_obj),
                 "gnn_obj_std": np.std(best_gnn_obj),
                 "gnn_violation_mean": np.mean(gnn_violations),
                 "gnn_violation_std": np.std(gnn_violations),
                 "gnn_time_mean": np.mean(gnn_times),
                 "gnn_time_std": np.std(gnn_times),
                 # for latex convenience
                 "percent_obj_string": f'{np.mean(best_gnn_obj) * 100:.3f}'
                                       f'\scriptsize$\pm${np.std(best_gnn_obj) * 100:.3f}',
                 "gnn_time_string": f'{np.mean(gnn_times):.3f}'
                                    f'\scriptsize$\pm${np.std(gnn_times):.3f}',
                 "vio_string": f'{np.mean(gnn_violations):.3f}'
                               f'\scriptsize$\pm${np.std(gnn_violations):.3f}'
                 }

    wandb.log(stat_dict)


if __name__ == '__main__':
    main()
