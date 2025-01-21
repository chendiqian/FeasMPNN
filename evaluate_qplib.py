import os

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from data.collate_func import collate_fn_lp_base
from data.transforms import GCNNorm
from data.utils import qp_obj
from models.feasible_unroll_model import FeasibleUnrollGNN
from models.base_hetero_gnn import BipartiteHeteroGNN, TripartiteHeteroGNN
from trainer import Trainer


@hydra.main(version_base=None, config_path='./config', config_name="evaluate")
def main(args: DictConfig):
    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.load(args.datapath).to(device)
    data = GCNNorm()(data) if 'gcn' in args.conv else data
    nulls = data.nulls.reshape(1, data.x_solution.shape[0], -1)
    data = collate_fn_lp_base([data])
    data.proj_matrix = nulls
    data.x_start = data.x_feasible

    # warmup and set dimensions
    ModelClass = TripartiteHeteroGNN if args.tripartite else BipartiteHeteroGNN
    gnn = ModelClass(conv=args.conv,
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
    model = FeasibleUnrollGNN(1, 1., args.ipm_eval_steps, gnn,
                              args.barrier_strength, args.tau, args.tau_scale).to(device)

    # warm up
    with torch.no_grad():
        for _ in range(10):
            _ = gnn(data, data.x_start)

    # end warming up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # prep pretrained model
    if args.modelpath is not None:
        model_list = [n for n in os.listdir(args.modelpath) if n.endswith('.pt')]
    else:
        model_list = [None]

    rel_obj_gap = []
    vios = []
    times = []
    abs_obj = []

    # begin evaluation
    for ckpt in model_list:
        if ckpt is not None:
            model.load_state_dict(torch.load(os.path.join(args.modelpath, ckpt), map_location=device))
        model.eval()

        # null space and x_feasible pre-process
        data = data.to(device)
        final_x, best_obj, obj_gaps, time_stamps, cos_sims = model.evaluation(data)
        times.append(time_stamps[-1])
        best_obj = best_obj.cpu().numpy()
        rel_obj_gap.append(best_obj)
        vios.append(Trainer.violate_per_batch(final_x, data).cpu().numpy())
        abs_obj.append(qp_obj(final_x, data))

        wandb.log({'gap': best_obj.mean(), 'vio': vios[-1].mean()})

    stat_dict = {"rel_obj_mean": np.mean(rel_obj_gap),
                 "rel_obj_std": np.std(rel_obj_gap),
                 "abs_obj_mean": np.mean(abs_obj),
                 "abs_obj_std": np.std(abs_obj),
                 "gnn_violation_mean": np.mean(vios),
                 "gnn_violation_std": np.std(vios),
                 "gnn_time_mean": np.mean(times),
                 "gnn_time_std": np.std(times)}

    wandb.log(stat_dict)


if __name__ == '__main__':
    main()
