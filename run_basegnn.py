import os
from functools import partial

import hydra
import copy
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_base
from data.transforms import GCNNorm
from data.prefetch_generator import BackgroundGenerator
from models.plain_gnn import BaseBipartiteHeteroGNN
from trainer import PlainGNNTrainer
from data.utils import save_run_config


@hydra.main(version_base=None, config_path='./config', config_name="run_base")
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
    collate_fn = partial(collate_fn_lp_base)
    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)],
                              batch_size=args.batchsize,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                            batch_size=args.val_batchsize,
                            shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):],
                             batch_size=args.val_batchsize,
                             shuffle=False,
                             collate_fn=collate_fn)

    best_val_objgaps = []
    test_objgaps = []
    test_violations = []

    for run in range(args.runs):
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
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=50 // args.eval_every,
                                                         min_lr=1.e-5)

        trainer = PlainGNNTrainer(args.losstype, args.coeff_obj, args.coeff_vio)

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss, train_vios = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)
            stats_dict = {'train_loss': train_loss,
                          'train_vios': train_vios,
                          'lr': scheduler.optimizer.param_groups[0]["lr"]}
            if epoch % args.eval_every == 0:
                val_obj_gap, val_vio = trainer.eval(BackgroundGenerator(val_loader, device, 4), model)

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
                stats_dict['val_vio'] = val_vio

            pbar.set_postfix(stats_dict)
            wandb.log(stats_dict)
        best_val_objgaps.append(trainer.best_objgap)

        model.load_state_dict(best_model)
        test_obj_gap, test_violation = trainer.eval(test_loader, model)
        test_objgaps.append(test_obj_gap)
        test_violations.append(test_violation)
        wandb.log({'test_obj_gap': test_obj_gap,
                   'test_violation': test_violation})

    wandb.log({
        'best_val_obj_gap': np.mean(best_val_objgaps),
        'test_obj_gap_mean': np.mean(test_objgaps),
        'test_obj_gap_std': np.std(test_objgaps),
        'test_violation_mean': np.mean(test_violations),
        'test_violation_std': np.std(test_violations),
    })


if __name__ == '__main__':
    main()
