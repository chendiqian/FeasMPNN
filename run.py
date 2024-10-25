import os
import copy

import hydra
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_bi
from data.transforms import GCNNorm
from data.prefetch_generator import BackgroundGenerator
from models.hetero_gnn import TripartiteHeteroGNN, BipartiteHeteroGNN
from models.cycle_model import CycleGNN
from trainer import Trainer
from data.utils import save_run_config


@hydra.main(version_base=None, config_path='./config', config_name="run")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath, transform=GCNNorm() if 'gcn' in args.conv else None)
    if args.debug:
        dataset = dataset[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)],
                              batch_size=args.batchsize,
                              shuffle=True,
                              collate_fn=collate_fn_lp_bi)
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                            batch_size=args.val_batchsize,
                            shuffle=False,
                            collate_fn=collate_fn_lp_bi)
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):],
                             batch_size=args.val_batchsize,
                             shuffle=False,
                             collate_fn=collate_fn_lp_bi)

    best_val_objgaps = []
    test_objgaps = []

    for run in range(args.runs):
        gnn = BipartiteHeteroGNN(conv=args.conv,
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
        model = CycleGNN(args.ipm_train_steps,
                         args.train_frac,
                         args.ipm_eval_steps,
                         gnn,
                         args.barrier_strength, args.tau, args.tau_scale).to(device)
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=70 // args.eval_every,
                                                         min_lr=1.e-5)

        trainer = Trainer(args.losstype, args.microbatch, args.coeff_l2, args.coeff_cos)

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss, train_cos_sims = trainer.train(BackgroundGenerator(train_loader, device, 2), model, optimizer)
            stats_dict = {'train_loss': train_loss,
                          'lr': scheduler.optimizer.param_groups[0]["lr"]}
            if epoch % args.eval_every == 0:
                half_objgap, val_obj_gap = trainer.eval(BackgroundGenerator(val_loader, device, 2), model)
                if scheduler is not None:
                    scheduler.step(val_obj_gap)

                if trainer.best_objgap > val_obj_gap:
                    trainer.patience = 0
                    trainer.best_objgap = val_obj_gap
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt:
                        torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
                else:
                    trainer.patience += 1

                if trainer.patience > (args.patience // args.eval_every + 1):
                    break

                stats_dict['val_obj_gap'] = val_obj_gap
                stats_dict['1/2_obj_gap'] = half_objgap

            pbar.set_postfix(stats_dict)
            # log the cossim, but not show them
            for idx, cossim in enumerate(train_cos_sims):
                stats_dict[f'train_cossim_{idx}'] = round(cossim, 3)
            wandb.log(stats_dict)
        best_val_objgaps.append(trainer.best_objgap)

        model.load_state_dict(best_model)
        _, test_obj_gap = trainer.eval(BackgroundGenerator(test_loader, device, 4), model)
        test_objgaps.append(test_obj_gap)
        wandb.log({'test_obj_gap': test_obj_gap})

    wandb.log({
        'best_val_obj_gap': np.mean(best_val_objgaps),
        'test_obj_gap_mean': np.mean(test_objgaps),
        'test_obj_gap_std': np.std(test_objgaps),
    })


if __name__ == '__main__':
    main()
