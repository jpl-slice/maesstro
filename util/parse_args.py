import argparse
import os
from warnings import warn

import torch

from .yaml_config_hook import yaml_config_hook


def parse_args(config=None, **kwargs):
    parser = argparse.ArgumentParser(description="MoCo")

    # parse config file first, then add arguments from config file
    config = "./config/default_config.yaml" if config is None else config
    parser.add_argument("--config", default=config)
    args, unknown = parser.parse_known_args()
    config = yaml_config_hook(args.config)

    # add arguments from `config` dictionary into parser, handling boolean args too
    bool_configs = [
        "multiprocessing_distributed",
        "teacher_weights",
        "student_weights",
        "cos",
        "mlp",
        "aug_plus",
        "use_wandb",
        "evaluate_mae_predictions",
        "wandb_run_id",
    ]
    for k, v in config.items():
        if k == "config":  # already added config earlier, so skip
            continue
        v = kwargs.get(k, v)
        if k in bool_configs:
            parser.add_argument(f"--{k}", default=v, type=str)
        elif k.lower() in ["seed", "pretrain_epoch_num"]:
            parser.add_argument(f"--{k}", default=v, type=int)
        else:
            parser.add_argument(f"--{k}", default=v, type=type(v))
    for k, v in kwargs.items():
        if k not in config:
            parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("--port", default=str(find_free_port()))

    # parse added arguments
    args, _ = parser.parse_known_args()
    for k, v in vars(args).items():
        if k in bool_configs and isinstance(v, str):
            if v.lower() in ["yes", "no", "true", "false", "none"]:
                exec(f'args.{k} = v.lower() in ["yes", "true"]')

    # Use built-in Python os utils to enable "~/path/to/file" or "$SCRATCH/path/to/file" in the command line
    args.data_path = os.path.expanduser(os.path.expandvars(args.data_path))
    args.val_data_path = os.path.expanduser(os.path.expandvars(args.val_data_path))
    args.output_dir = os.path.expanduser(os.path.expandvars(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.ngpus_per_node != torch.cuda.device_count():
        warn(
            f"WARNING: Specified ngpus_per_node doesn't match torch.cuda.device_count()"
            f" ({args.ngpus_per_node} != {torch.cuda.device_count()}). "
        )
        # ? Should we set args.ngpus_per_node = torch.cuda.device_count()?

    if args.world_size == -1:
        args.world_size = int(
            os.environ.get(
                "SLURM_NTASKS",
                os.environ.get("WORLD_SIZE", args.ngpus_per_node * args.nodes),
            )
        )

    return args


def find_free_port():
    # taken from https://github.com/ShigekiKarita/pytorch-distributed-slurm-example/blob/master/main_distributed.py
    import socket

    s = socket.socket()
    s.bind(("", 0))  # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.
