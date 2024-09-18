# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import datetime
import json
import os
import time

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn

import models_mae
import util.misc as misc
from data_utils import create_data_loader
from engine_pretrain import eval_one_epoch, train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import set_manual_seed
from util.parse_args import parse_args
from util.tensorboard_wandb_logging import setup_tensorboard_and_wandb


def main(args):
    misc.init_distributed_mode(0, args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    set_manual_seed(args.seed)  # fix the seed for reproducibility

    cudnn.benchmark = True

    train_loader, val_loader = create_data_loader(args)

    # define the model
    try:
        model = models_mae.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            img_size=args.input_size,
            in_chans=args.input_channels,
        )
    except KeyError:
        model = models_mae.mae_builder(
            args.model,
            norm_pix_loss=args.norm_pix_loss,
            img_size=args.input_size,
            in_chans=args.input_channels,
        )
    args.num_params = sum(p.numel() for p in model.parameters())
    print(
        f"Created {args.model} (img_size={args.input_size}, "
        f"in_chans={args.input_channels}) with {args.num_params} parameters"
    )

    model.to(args.device)
    model_without_ddp = model
    args.model_without_ddp = model_without_ddp

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    # for older version of timm, use add_weight_decay(model, weight_decay, skip_list=())
    param_groups = optim_factory.param_groups_weight_decay(
        model_without_ddp, args.weight_decay
    )
    set_lr_from_batch_size(args)  # set args.lr = base_lr * eff_batch_size / 256
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    args.optimizer = optimizer.__class__.__name__  # Add this so that we can log in W&B
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args, model_without_ddp, optimizer, loss_scaler)

    if args.global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = setup_tensorboard_and_wandb(args, len(train_loader.dataset))
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        ep_start_time = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            args, model, train_loader, optimizer, epoch, loss_scaler, log_writer
        )
        # Save checkpoint and run eval if needed
        is_last_epoch = (epoch + 1) == args.epochs
        if args.output_dir and (epoch % 10 == 0 or is_last_epoch):
            # save_model handles multi-processing; only saves on the main process to avoid overwriting
            ckpt_path = misc.save_model(
                args, epoch, model, model_without_ddp, optimizer, loss_scaler
            )
            if args.evaluate_mae_predictions and val_loader is not None:
                val_stats = eval_one_epoch(args, model, val_loader, epoch, log_writer)
                train_stats.update(val_stats)
            if misc.is_main_process() and is_last_epoch and args.use_wandb:
                try:
                    import wandb

                    wandb.save(ckpt_path)
                except Exception as e:
                    print(f"Tried saving {ckpt_path} to W&B but failed with {e}.")

        log_stats = {**train_stats, "epoch": epoch}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                ep_ = epoch * 1000
                log_writer.add_scalar("epoch_time", time.time() - ep_start_time, ep_)
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def set_lr_from_batch_size(args):
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    args.eff_batch_size = eff_batch_size
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print(
        f"Effective batch size: {eff_batch_size}\n"
        f"Base LR: {(args.blr):.2e}\n"
        f"Actual LR (blr * eff_batch / 256): {args.lr:.2e}\n"
        f"Accumulative grad iterations: {args.accum_iter}\n"
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)
