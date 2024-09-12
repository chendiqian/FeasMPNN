import os
import argparse
import logging

import copy
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import wandb

from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_bi
from data.transforms import GCNNorm
from models.hetero_gnn import BipartiteHeteroGNN
from models.cycle_model import CycleGNN
from trainer import MultiGPUTrainer
from data.utils import save_run_config

logging.basicConfig(level=logging.INFO, format="{asctime} - {message}", style="{", datefmt="%Y-%m-%d %H:%M:%S")


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', default=False, action='store_true')

    # training dynamics
    parser.add_argument('--losstype', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--ckpt', default=False, action='store_true')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=300)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--val_batchsize', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--coeff_l2', type=float, default=0.1, help='balance between L2loss and cos loss')
    parser.add_argument('--coeff_cos', type=float, default=1., help='balance between L2loss and cos loss')

    # model related
    parser.add_argument('--ipm_train_steps', type=int, default=8)
    parser.add_argument('--train_frac', type=float, default=1.)
    parser.add_argument('--ipm_eval_steps', type=int, default=32)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--tau_scale', type=float, default=0.5)
    parser.add_argument('--plain_xstarts', default=False, action='store_true')
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--conv', type=str, default='gcnconv')
    parser.add_argument('--heads', type=int, default=1, help='for GAT only')
    parser.add_argument('--concat', default=False, action='store_true', help='for GAT only')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_encode_layers', type=int, default=2)
    parser.add_argument('--num_conv_layers', type=int, default=8)
    parser.add_argument('--num_pred_layers', type=int, default=3)
    parser.add_argument('--hid_pred', type=int, default=-1)
    parser.add_argument('--num_mlp_layers', type=int, default=1)
    parser.add_argument('--norm', type=str, default='graphnorm')  # empirically better

    return parser.parse_args()


def run(rank, dataset, world_size, log_folder_name, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    train_set = dataset[:int(len(dataset) * 0.8)]
    val_set = dataset[int(len(dataset) * 0.8): int(len(dataset) * 0.9)]
    test_set = dataset[int(len(dataset) * 0.9):]

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_set,
                              num_workers=args.num_workers,
                              batch_size=args.batchsize // world_size,
                              collate_fn=collate_fn_lp_bi,
                              sampler=train_sampler)
    val_loader = DataLoader(val_set,
                            num_workers=args.num_workers,
                            batch_size=args.val_batchsize // world_size,
                            sampler=val_sampler,
                            collate_fn=collate_fn_lp_bi)
    test_loader = DataLoader(test_set,
                             num_workers=args.num_workers,
                             batch_size=args.val_batchsize // world_size,
                             sampler=test_sampler,
                             collate_fn=collate_fn_lp_bi)

    if rank == 0:
        wandb.init(project=args.wandbproject,
                   name=args.wandbname if args.wandbname else None,
                   mode="online" if args.use_wandb else "disabled",
                   config=vars(args),
                   entity="chendiqian")  # use your own entity

        best_val_objgaps = []
        test_objgaps = []

    torch.cuda.set_device(rank)
    for run in range(args.runs):
        torch.cuda.empty_cache()
        dist.barrier()

        gnn = BipartiteHeteroGNN(conv=args.conv,
                                 head=args.heads,
                                 concat=args.concat,
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
                         args.tau, args.tau_scale).to(rank)
        model = DistributedDataParallel(model, device_ids=[rank])
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=70 // args.eval_every,
                                                         min_lr=1.e-5)

        trainer = MultiGPUTrainer(args.losstype, args.microbatch, args.coeff_l2, args.coeff_cos)

        for epoch in range(args.epoch):
            train_sampler.set_epoch(epoch)
            train_loss, train_cos_sims = trainer.train(train_loader, model, optimizer, rank)

            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(train_cos_sims, op=dist.ReduceOp.AVG)

            if rank == 0:
                stats_dict = {'epoch': epoch,
                              'train_loss': train_loss.item(),
                              'train_cos_sim_0': train_cos_sims[0].item(),
                              f'train_cos_sim_{train_cos_sims.shape[0] - 1}': train_cos_sims[-1].item(),
                              'lr': scheduler.optimizer.param_groups[0]["lr"]}

            if epoch % args.eval_every == 0:
                val_obj_gap = trainer.eval(val_loader, model.module, rank)
                dist.all_reduce(val_obj_gap, op=dist.ReduceOp.AVG)
                val_obj_gap = val_obj_gap.item()

                if scheduler is not None:
                    scheduler.step(val_obj_gap)

                if trainer.best_objgap > val_obj_gap:
                    trainer.patience = 0
                    trainer.best_objgap = val_obj_gap
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt:
                        torch.save(model.module.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
                else:
                    trainer.patience += 1

                if trainer.patience > (args.patience // args.eval_every + 1):
                    break

                if rank == 0:
                    stats_dict['val_obj_gap'] = val_obj_gap

            if rank == 0:
                infos = ', '.join([k + f':{v:.6f}' for k, v in stats_dict.items()])
                logging.info(infos)
                wandb.log(stats_dict)

        dist.barrier()
        model.load_state_dict(best_model)
        test_obj_gap = trainer.eval(test_loader, model.module, rank)
        dist.all_reduce(test_obj_gap, op=dist.ReduceOp.AVG)
        dist.barrier()
        test_obj_gap = test_obj_gap.item()

        if rank == 0:
            best_val_objgaps.append(trainer.best_objgap)
            test_objgaps.append(test_obj_gap)
            logging.info(f"run: {run} finished! test_obj_gap: {test_obj_gap}")
            wandb.log({'test_obj_gap': test_obj_gap})

    if rank == 0:
        wandb.log({
            'best_val_obj_gap': np.mean(best_val_objgaps),
            'test_obj_gap_mean': np.mean(test_objgaps),
            'test_obj_gap_std': np.std(test_objgaps),
        })

    dist.barrier()
    # at the very end
    dist.destroy_process_group()


if __name__ == '__main__':
    args = args_parser()
    log_folder_name = save_run_config(args)

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert available_gpus > 1, "This running file for multi gpu usage only!!!!"

    dataset = LPDataset(args.datapath, transform=GCNNorm() if 'gcn' in args.conv else None)
    mp.spawn(run, args=(dataset, available_gpus, log_folder_name, args), nprocs=available_gpus, join=True)
