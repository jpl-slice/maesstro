# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
import time
from typing import Iterable

import torch

import util.lr_sched as lr_sched
import util.misc as misc
from data_utils.llc4320_sst import Tstd_llc
from util.metrics import initialize_metrics
from util.tensorboard_wandb_logging import plot_samples_tensorboard

max_num_samples_to_plot = 10


Tmin_jinbo = 9.883099555969238
Tmax_jinbo = 23.177156448364258
Tmean_jinbo = 16.894472122192383


def train_one_epoch(
    args,
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss_scaler,
    log_writer=None,
    plot_samples=True,
):
    global max_num_samples_to_plot
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    torch_metrics = initialize_metrics(args, prefix="train_")

    optimizer.zero_grad()

    if log_writer is not None:
        print("Tensorboard log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        max_num_samples_to_plot = min(max_num_samples_to_plot, samples.shape[0])
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % args.accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # skip iteration if any of the input samples are nan
        if torch.isnan(samples).any():
            print("nan samples detected, exiting...")
            sys.exit(1)

        samples = samples.to(args.device, non_blocking=True)

        # https://github.com/facebookresearch/mae/issues/42
        with torch.cuda.amp.autocast(enabled=True):
            loss, r2, outputs, masks = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        r2_value = r2.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / args.accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % args.accum_iter == 0,
            # https://github.com/facebookresearch/mae/issues/42#issuecomment-1327427371
            clip_grad=True,
        )
        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(r2=r2_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_mean = misc.all_reduce_mean(loss_value)
        r2_mean = misc.all_reduce_mean(r2_value)

        # update torch_metrics using only masked pixels from each image
        # select masked pixels and flatten entire batch into single list of unmasked pixels
        m_ = model.module if hasattr(model, "module") else model
        m = masks.unsqueeze(-1).repeat(
            1, 1, m_.patch_embed.patch_size[0] ** 2 * m_.input_channels
        )
        m = m_.unpatchify(m).detach()
        o = m_.unpatchify(outputs).detach()
        masked_pred_pixels = o[m == 1].flatten()
        masked_truth_pixels = samples[m == 1].flatten()
        torch_metrics.update(preds=masked_pred_pixels, target=masked_truth_pixels)
        rmse = loss_mean**0.5 * Tstd_llc

        if log_writer is not None and (data_iter_step + 1) % args.accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            ep = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_metrics(
                log_writer, metric_logger, lr, loss_mean, r2_mean, rmse, "train", ep
            )
    # plot samples at the end of the epoch
    plot_samples = plot_samples and (epoch % 5 == 0)  # only plot every 5 epochs
    if plot_samples and misc.is_main_process() and log_writer is not None:
        _plot_samples(model, epoch, log_writer, samples, outputs, masks, tag="train")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    output_dict = {f"train_{k}": m.global_avg for k, m in metric_logger.meters.items()}
    # add torch_metrics to output_dict
    torch_metrics_dict = {k: v.item() for k, v in torch_metrics.compute().items()}
    torch_metrics.reset()
    torch_metrics_dict["train_RMSE (℃)"] = loss_mean**0.5 * Tstd_llc
    output_dict.update(torch_metrics_dict)
    print("Averaged stats:", output_dict)
    if log_writer is not None:
        add_scalars(log_writer, output_dict, "train", int((1 + epoch) * 1000))
    return output_dict


def log_metrics(writer, logger, lr, loss_mean, r2_mean, rmse, prefix, epoch):
    log = lambda k, v: writer.add_scalar(f"{prefix}/{k}", v, epoch)
    log("loss", loss_mean)
    log("r2", r2_mean)
    log("RMSE (℃)", rmse)

    if prefix == "train":
        try:
            log("iter_time", logger.meters["iter_time"].value)
            log("data_time", logger.meters["data_time"].value)
        except:
            # can't log iter_time and date_time during the first iteration
            # because they are not yet defined
            pass
        writer.add_scalar("lr", lr, epoch)


def add_scalars(writer, out_dict, prefix, epoch):
    """Since log_writer.add_scalars doesn't work with W&B,
    let's write our own function to do the same thing.
    """
    for k, v in out_dict.items():
        k = k.split("_")[-1]
        writer.add_scalar(f"{prefix}/averaged/{k}", v, epoch)


def _plot_samples(model, epoch, writer, samples, preds, masks, tag=""):
    t_start = time.time()  # use this to calculate time taken

    m = model.module if hasattr(model, "module") else model

    # take only the first max_num_samples_to_plot samples
    samples = samples[:max_num_samples_to_plot]
    masks = masks[:max_num_samples_to_plot]
    preds = preds[:max_num_samples_to_plot]

    masks = masks.unsqueeze(-1).repeat(
        1, 1, m.patch_embed.patch_size[0] ** 2 * m.input_channels
    )
    masks = torch.einsum("nchw->nhwc", m.unpatchify(masks)).detach().cpu()
    preds = torch.einsum("nchw->nhwc", m.unpatchify(preds)).detach().cpu()

    # Plot and add to tensorboard
    plot_samples_tensorboard(epoch, writer, samples, preds, masks, tag_prefix=tag)
    print(f"Plotting time taken: {time.time() - t_start:.2f} seconds")


def eval_one_epoch(
    args,
    model: torch.nn.Module,
    loader: Iterable,
    epoch: int,
    log_writer=None,
    plot_samples=True,
):
    global max_num_samples_to_plot
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "[VAL] Epoch: [{}]".format(epoch)
    print_freq = 20

    torch_metrics = initialize_metrics(args, prefix="val_")

    if log_writer is not None:
        print("Tensorboard log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(
        metric_logger.log_every(loader, print_freq, header)
    ):
        max_num_samples_to_plot = min(max_num_samples_to_plot, samples.shape[0])

        # skip iteration if any of the input samples are nan
        if torch.isnan(samples).any():
            print("nan samples detected, exiting...")
            sys.exit(1)

        samples = samples.to(args.device, non_blocking=True)

        with torch.no_grad():
            loss, r2, outputs, masks = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        r2_value = r2.item()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        loss_mean = misc.all_reduce_mean(loss_value)
        r2_mean = misc.all_reduce_mean(r2_value)

        # update torch_metrics using only masked pixels from each image
        # select masked pixels and flatten entire batch into single list of unmasked pixels
        m_ = model.module if hasattr(model, "module") else model
        m = masks.unsqueeze(-1).repeat(
            1, 1, m_.patch_embed.patch_size[0] ** 2 * m_.input_channels
        )
        m = m_.unpatchify(m).detach()
        o = m_.unpatchify(outputs).detach()
        masked_pred_pixels = o[m == 1].flatten()
        masked_truth_pixels = samples[m == 1].flatten()
        N = samples.shape[0]
        torch_metrics.update(
            preds=masked_pred_pixels,
            target=masked_truth_pixels,
        )
        rmse = (loss_mean**0.5) * Tstd_llc
        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            ep = int((data_iter_step / len(loader) + epoch) * 1000)
            log_metrics(
                log_writer, metric_logger, 0, loss_mean, r2_mean, rmse, "val", ep
            )
    # plot samples at the end of the epoch
    # only plot every 20 epochs
    # plot_samples = plot_samples and (epoch % 20 == 0)
    if plot_samples and misc.is_main_process() and log_writer is not None:
        _plot_samples(model, epoch, log_writer, samples, outputs, masks, tag="val")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    output_dict = {f"val_{k}": m.global_avg for k, m in metric_logger.meters.items()}
    # add torch_metrics to output_dict
    torch_metrics_dict = {k: v.item() for k, v in torch_metrics.compute().items()}
    torch_metrics.reset()
    torch_metrics_dict["val_RMSE (℃)"] = loss_mean**0.5 * Tstd_llc
    output_dict.update(torch_metrics_dict)
    print("Averaged stats (val):", output_dict)
    if log_writer is not None:
        add_scalars(log_writer, output_dict, "val", int((1 + epoch) * 1000))
    return {f"val_{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
