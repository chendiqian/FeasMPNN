import hydra
import numpy as np
import wandb
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from data.dataset import LPDataset
from data.utils import sync_timer, recover_qp_from_data
from qpsolvers import solve_qp


@hydra.main(version_base=None, config_path='./config', config_name="evaluate")
def main(args: DictConfig):
    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")

    dataset = LPDataset(args.datapath)[-100:]

    cvxopt_time = []
    osqp_time = []

    pbar = tqdm(dataset)
    for data in pbar:
        P, q, A, b, G, h, lb, ub = recover_qp_from_data(data, np.float64)

        start_t = sync_timer()
        solution = solve_qp(P, q, G, h, A, b, lb=lb, ub=ub, solver="cvxopt")
        end_t = sync_timer()
        cvxopt_time.append(end_t - start_t)

        start_t = sync_timer()
        solution = solve_qp(P, q, G, h, A, b, lb=lb, ub=ub, solver="osqp")
        end_t = sync_timer()
        osqp_time.append(end_t - start_t)

        stat_dict = {'cvxopt': cvxopt_time[-1],
                     'osqp': osqp_time[-1]}

        wandb.log(stat_dict)
        pbar.set_postfix(stat_dict)

    stat_dict = {"cvxopt_mean": np.mean(cvxopt_time),
                 "cvxopt_std": np.std(cvxopt_time),
                 "osqp_mean": np.mean(osqp_time),
                 "osqp_std": np.std(osqp_time)}

    wandb.log(stat_dict)


if __name__ == '__main__':
    main()
