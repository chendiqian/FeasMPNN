import os
import logging

import hydra
import copy
import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import wandb
from omegaconf import DictConfig, OmegaConf

from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_base
from data.transforms import GCNNorm
from models.hetero_gnn import BipartiteHeteroGNN
from models.ipm_model import IPMGNN
from trainer import MultiGPUIPMTrainer
from data.utils import save_run_config

logging.basicConfig(level=logging.INFO, format="{asctime} - {message}", style="{", datefmt="%Y-%m-%d %H:%M:%S")


@hydra.main(version_base=None, config_path='./config', config_name="run_ipm")
def main(args: DictConfig):
    world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
    rank = int(os.environ['RANK'])  # Rank of the current process
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size > 1, "This running file for multi gpu usage only!!!!"

    dist.init_process_group(backend="nccl")

    if rank == 0:
        log_folder_name = save_run_config(args)

    dataset = LPDataset(args.datapath, transform=GCNNorm() if 'gcn' in args.conv else None)
    if args.debug:
        dataset = dataset[:20]

    train_set = dataset[:int(len(dataset) * 0.8)]
    val_set = dataset[int(len(dataset) * 0.8): int(len(dataset) * 0.9)]
    test_set = dataset[int(len(dataset) * 0.9):]

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_set,
                              num_workers=args.num_workers,
                              batch_size=args.batchsize // world_size,
                              collate_fn=collate_fn_lp_base,
                              sampler=train_sampler)
    val_loader = DataLoader(val_set,
                            num_workers=args.num_workers,
                            batch_size=args.val_batchsize // world_size,
                            sampler=val_sampler,
                            collate_fn=collate_fn_lp_base)
    test_loader = DataLoader(test_set,
                             num_workers=args.num_workers,
                             batch_size=args.val_batchsize // world_size,
                             sampler=test_sampler,
                             collate_fn=collate_fn_lp_base)

    if rank == 0:
        wandb.init(project=args.wandb.project,
                   name=args.wandb.name if args.wandb.name else None,
                   mode="online" if args.wandb.enable else "disabled",
                   config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
                   entity="chendiqian")  # use your own entity

        best_val_objgaps = []
        test_objgaps = []

    torch.cuda.set_device(local_rank)
    for run in range(args.runs):
        torch.cuda.empty_cache()
        dist.barrier()

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
        model = IPMGNN(args.ipm_train_steps,
                       args.ipm_eval_steps,
                       gnn).to(local_rank)
        model = DistributedDataParallel(model, device_ids=[local_rank])
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=70 // args.eval_every,
                                                         min_lr=1.e-5)

        trainer = MultiGPUIPMTrainer(args.losstype, args.microbatch)

        for epoch in range(args.epoch):
            train_sampler.set_epoch(epoch)
            train_loss = trainer.train(train_loader, model, optimizer, local_rank)

            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)

            if rank == 0:
                stats_dict = {'epoch': epoch,
                              'train_loss': train_loss.item(),
                              'lr': scheduler.optimizer.param_groups[0]["lr"]}

            if epoch % args.eval_every == 0:
                val_obj_gap = trainer.eval(val_loader, model.module, local_rank)
                dist.all_reduce(val_obj_gap, op=dist.ReduceOp.AVG)
                val_obj_gap = val_obj_gap.item()

                if scheduler is not None:
                    scheduler.step(val_obj_gap)

                if trainer.best_objgap > val_obj_gap:
                    trainer.patience = 0
                    trainer.best_objgap = val_obj_gap
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt and rank == 0:
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
        test_obj_gap = trainer.eval(test_loader, model.module, local_rank)
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
    main()
