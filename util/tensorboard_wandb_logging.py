import io
import os
import shutil
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import yaml
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import ToTensor

try:
    import wandb
except:
    pass  # Don't worry if we can't import W&B; just ignore and log to Tensorboard


def setup_tensorboard_and_wandb(args, num_images):
    writer = None
    if args.global_rank == 0:
        # if wandb is imported, do wandb.init
        # must call this BEFORE creating summary writer
        run_name = get_run_name(args, num_images)

        # Use log dir with run_name as the primary output_dir from now on:
        args.output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(args.output_dir, exist_ok=True)

        # save current args as a yaml file in args.output_dir
        config_backup_path = save_config_backup(args)

        args.global_step = 0
        if "wandb" in sys.modules and args.use_wandb:
            # note: w&b will crash if the provided run_id doesn't exist since there's nothing to resume from
            wandb_resume = None if args.wandb_run_id is None else "must"
            # set up config_dict to log:
            config_dict = vars(args)
            config_dict.pop("model_without_ddp", None)  # remove "model_without_ddp"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                resume=wandb_resume,
                id=args.wandb_run_id,
                config=config_dict,
                sync_tensorboard=True,
                save_code=True,
            )
            args.global_step = wandb.run.step
            wandb.save(config_backup_path)
            wandb.run.log_code(".")
        writer = SummaryWriter(log_dir=args.output_dir)
    return writer


def get_run_name(args, num_images):
    run_name = (
        f"MAE|{args.dataset}|"
        f"N{num_images}|Sz{args.input_size}|C{args.input_channels}|"
        f"Sd{args.seed}|"
        f"B{args.batch_size * args.world_size}|"
        f"{args.start_epoch}"
        f"-{args.epochs}|"
        f"M{args.model}|"
        f"MR{args.mask_ratio}|{args.optimizer}|"
        f"{datetime.now().strftime('%y%m%d-%H%M%S')}"  # timestamp
    )
    return run_name


def save_config_backup(args):
    config_backup_path = os.path.join(
        args.output_dir, f"config_{datetime.now().strftime('%y%m%d-%H%M%S')}.yaml"
    )
    current_config_dict = vars(args)
    # load configs from the provided config.yaml, which may be
    # different from current_config_dict due to command line args:
    with open(args.config, "r") as f:
        config_file_dict = yaml.load(f, Loader=yaml.FullLoader)

    # for every item in config_file_dict, make sure that the values match that of current_config_dict
    backup_config_dict = dict()
    for key, value in config_file_dict.items():
        backup_config_dict[key] = current_config_dict.get(key, value)

    with open(config_backup_path, "w") as f:
        # write the dictionary to a formatted yaml file
        yaml.dump(backup_config_dict, f, default_flow_style=False)

    return config_backup_path


def plot_samples_tensorboard(epoch, log_writer, samples, outputs, masks, tag_prefix=""):
    fig = plt.figure(figsize=(6, 15))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(samples.shape[0], 3),  # creates 2x2 grid of axes
        axes_pad=0.02,  # pad between axes in inch.
    )

    for row in range(0, samples.shape[0]):
        # for each row, column 1 = input, column 2 = input * (1 - mask), column 3 = output
        input_ = torch.einsum("chw->hwc", samples[row]).detach().cpu()
        # adjust the color scale to start from the minimum value
        vmin, vmax = input_.min(), input_.max()
        grid[row * 3].imshow(input_, vmin=vmin, vmax=vmax)

        # set masked out pixels to masked_value
        # make sure that masked value is always less than the minimum value in input_
        masked_value = 0 if vmin >= 0.1 else -2 * np.abs(vmin)
        masked_input = input_.clone()
        masked_input[masks[row] == 1] = masked_value
        grid[row * 3 + 1].imshow(masked_input, vmin=vmin, vmax=vmax)

        # for the third col, replace model predictions on visible patches with the original
        # this is because MAE predicts noise on visible patches;
        # loss function isn't conditioned on visible patches.
        outputs[row][masks[row] == 0] = input_[masks[row] == 0].type(outputs.dtype)
        ax = grid[row * 3 + 2]
        im = ax.imshow(outputs[row], vmin=vmin, vmax=vmax)

        # add colorbar to end of column 3
        cax = inset_axes(
            ax,
            width="8%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.10, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        plt.colorbar(im, cax=cax)

        # turn off axes
        grid[row * 3].axis("off")
        grid[row * 3 + 1].axis("off")
        grid[row * 3 + 2].axis("off")

    plt.tight_layout()

    # Save figure to Tensorboard
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    tag = "samples" if tag_prefix == "" else f"{tag_prefix}_samples"
    log_writer.add_image(tag, image, epoch)

    plt.close(fig)
