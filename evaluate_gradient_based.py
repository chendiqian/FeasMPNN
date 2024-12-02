import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.collate_func import collate_fn_lp_bi
from data.dataset import LPDataset
from data.utils import recover_qp_from_data
from models.gradient_based_iter import GradSolver
from trainer import Trainer


@hydra.main(version_base=None, config_path='./config', config_name="evaluate_grad")
def main(args: DictConfig):
    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath)[-100:]
    if args.debug:
        dataset = dataset[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = DataLoader(dataset,
                            batch_size=args.batchsize,
                            shuffle=False,
                            collate_fn=collate_fn_lp_bi)

    # warmup and set dimensions
    model = GradSolver(num_eval_steps=args.ipm_eval_steps,
                       barrier_strength=args.barrier_strength,
                       init_tau=args.tau,
                       tau_scale=args.tau_scale).to(device)

    # begin evaluation
    model.eval()

    gaps = []
    vios = []
    times = []
    pbar = tqdm(dataloader)
    for data in pbar:
        # null space and x_feasible pre-process
        P, q, A, b, G, h, lb, ub = recover_qp_from_data(data)
        # P[np.diag(P) == 0.] += 1.e-6

        # use the property of convex quadratic function
        # f = 0.5 * (x + d) @ P @ (x + d) + q.dot(x + d) + lamb * |A(x + d) - b| ^ 2
        # 1st order: x + d = inv(P + 2 * lamb * A.T @ A) @ (2 * lamb * A.T @ A - q)
        rhs = np.linalg.pinv(P + 2 * args.lamb * A.T @ A) @ (2 * args.lamb * A.T @ b - q)
        rhs = torch.from_numpy(rhs).float().to(device)

        data = data.to(device)
        final_x, best_obj, _, time_stamps = model.evaluation(data, rhs)

        times.append(time_stamps[-1])
        best_obj = best_obj.cpu().numpy()
        gaps.append(best_obj)
        vios.append(Trainer.violate_per_batch(final_x, data).cpu().numpy())

        stat_dict = {'gap': best_obj.mean(),
                     'vio': vios[-1].mean(),
                     'solve_time': time_stamps[-1]}
        pbar.set_postfix(stat_dict)
        wandb.log(stat_dict)

    gaps = np.concatenate(gaps, axis=0)
    vios = np.concatenate(vios)

    stat_dict = {"obj_mean": np.mean(gaps),
                 "obj_std": np.std(gaps),
                 "violation_mean": np.mean(vios),
                 "violation_std": np.std(vios),
                 "time_mean": np.mean(times),
                 "time_std": np.std(times)}

    wandb.log(stat_dict)


if __name__ == '__main__':
    main()
