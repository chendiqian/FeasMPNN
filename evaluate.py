import os
import time

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from scipy.linalg import null_space
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.collate_func import collate_fn_lp_bi
from data.dataset import LPDataset
from data.transforms import GCNNorm
from data.utils import recover_qp_from_data
from models.feasible_unroll_model import FeasibleUnrollGNN
from models.base_hetero_gnn import BipartiteHeteroGNN, TripartiteHeteroGNN
from solver.linprog_ip import _ip_hsd_feas
from trainer import Trainer


@hydra.main(version_base=None, config_path='./config', config_name="evaluate")
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
                            collate_fn=collate_fn_lp_bi)

    best_gnn_obj = []
    gnn_timsteps = []
    gnn_times = []
    preprocess_times = []
    total_times = []
    gnn_violations = []

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
        data = next(iter(dataloader)).to(device)
        for _ in range(10):
            _ = gnn(data, data.x_start)

    _, _, A, b, _, _, _, _ = recover_qp_from_data(data.to('cpu'))
    for _ in range(20):
        _ = null_space(A)
        _ = _ip_hsd_feas(A, b, np.zeros(A.shape[1]), 0.,
                         alpha0=0.9999999, beta=0.1,
                         maxiter=5, tol=1.e-3, sparse=True,
                         lstsq=False, sym_pos=True, cholesky=None,
                         pc=True, ip=True, permc_spec='MMD_AT_PLUS_A',
                         rand_start=False)

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
        prep_times = []
        pbar = tqdm(dataloader)
        for data in pbar:
            # null space and x_feasible pre-process
            P, q, A, b, G, h, lb, ub = recover_qp_from_data(data)
            t0 = time.time()
            _ = null_space(A)
            t1 = time.time()
            null_time = t1 - t0
            _ = _ip_hsd_feas(A, b, np.zeros(A.shape[1]), 0.,
                             alpha0=0.9999999, beta=0.1,
                             maxiter=5, tol=1.e-3, sparse=True,
                             lstsq=False, sym_pos=True, cholesky=None,
                             pc=True, ip=True, permc_spec='MMD_AT_PLUS_A',
                             rand_start=False)
            t2 = time.time()
            feas_time = t2 - t1
            prep_times.append(t2 - t0)

            data = data.to(device)
            final_x, best_obj, obj_gaps, time_stamps, cos_sims = model.evaluation(data)
            gnn_timsteps.append(time_stamps)
            times.append(time_stamps[-1])
            best_obj = best_obj.cpu().numpy()
            gaps.append(best_obj)
            vios.append(Trainer.violate_per_batch(final_x, data).cpu().numpy())

            stat_dict = {'gap': best_obj.mean(),
                         'vio': vios[-1].mean(),
                         'feas_time': feas_time, 'null_time': null_time, 'gnn_time': time_stamps[-1]}
            pbar.set_postfix(stat_dict)
            wandb.log(stat_dict)
        gaps = np.concatenate(gaps, axis=0)
        vios = np.concatenate(vios)
        best_gnn_obj.append(np.mean(gaps))
        gnn_violations.append(np.mean(vios))
        gnn_times.append(np.mean(times))
        preprocess_times.append(np.mean(prep_times))
        total_times.append(preprocess_times[-1] + gnn_times[-1])

    time_per_step_gnn = [i[-1] / args.ipm_eval_steps for i in gnn_timsteps]

    stat_dict = {"gnn_obj_mean": np.mean(best_gnn_obj),
                 "gnn_obj_std": np.std(best_gnn_obj),
                 "gnn_violation_mean": np.mean(gnn_violations),
                 "gnn_violation_std": np.std(gnn_violations),
                 "gnn_time_per_step_mean": np.mean(time_per_step_gnn),
                 "gnn_time_per_step_std": np.std(time_per_step_gnn),
                 "total_time_mean": np.mean(total_times),
                 "total_time_std": np.std(total_times),
                 "prep_time_mean": np.mean(preprocess_times),
                 "prep_time_std": np.std(preprocess_times),
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
