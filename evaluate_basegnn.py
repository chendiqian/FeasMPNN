import os

import hydra
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from data.transforms import GCNNorm
from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_bi
from data.utils import sync_timer
from models.plain_gnn import BaseBipartiteHeteroGNN


@hydra.main(version_base=None, config_path='./config', config_name="eval_base")
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
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn_lp_bi)

    gnn_times = []
    best_gnn_obj = []
    gnn_violations = []

    # warmup and set dimensions
    model = BaseBipartiteHeteroGNN(conv=args.conv,
                                   head=args.gat.heads,
                                   concat=args.gat.concat,
                                   hid_dim=args.hidden,
                                   num_encode_layers=args.num_encode_layers,
                                   num_conv_layers=args.num_conv_layers,
                                   num_pred_layers=args.num_pred_layers,
                                   hid_pred=args.hid_pred,
                                   num_mlp_layers=args.num_mlp_layers,
                                   norm=args.norm).to(device)
    with torch.no_grad():
        data = next(iter(dataloader)).to(device)
        for _ in range(20):
            _ = model(data, data.x_start)

    for ckpt in [n for n in os.listdir(args.modelpath) if n.endswith('.pt')]:
        model.load_state_dict(torch.load(os.path.join(args.modelpath, ckpt), map_location=device))
        model.eval()

        gaps = []
        vios = []

        pbar = tqdm(dataloader)
        for data in pbar:
            data = data.to(device)
            t1 = sync_timer()
            prediction, obj_gap, vio = model.evaluation(data)
            t2 = sync_timer()
            obj_gap = obj_gap.cpu().numpy()
            vio = vio.cpu().numpy()
            vios.append(vio)
            gaps.append(obj_gap)
            gnn_times.append(t2 - t1)

            pbar.set_postfix({'obj_gap': obj_gap.mean(), 'violation': vio.mean()})

        best_gnn_obj.append(np.mean(np.concatenate(gaps)))
        gnn_violations.append(np.mean(np.concatenate(vios)))

    stat_dict = {"obj_mean": np.mean(best_gnn_obj),
                 "obj_std": np.std(best_gnn_obj),
                 "vio_mean": np.mean(gnn_violations),
                 "vio_std": np.std(gnn_violations),
                 "gnn_time_mean": np.mean(gnn_times),
                 "gnn_time_std": np.std(gnn_times)}

    wandb.log(stat_dict)


if __name__ == '__main__':
    main()
