import os
import sys

import torch
import torchvision.transforms as transforms

from data_utils.llc4320_sst import (
    SSTDatasetSingleNumpyTile,
    Tmean_jinbo,
    Tmean_llc,
    Tstd_jinbo,
    Tstd_llc,
    get_sst_data_dir,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util.misc as misc

mean_std_map = {
    "LLC4320_SST_GLOBAL": (Tmean_llc, Tstd_llc),
    "LLC4320_SST_JINBO": (Tmean_jinbo, Tstd_jinbo),
}


# This function was created to handle multiple datasets provided through
# `args.dataset`, but we've only implemented the LLC4320 dataset here.
def create_data_loader(args):
    """Creates data loaders for training and validation datasets based on the provided arguments.
    Also handles the data samplers for distributed training.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - dataset (str): The name of the dataset.
            - input_size (int): The size of the input images.
            - Other attributes required by `get_dataset_from_name` and `get_loader`.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - val_loader (DataLoader or None): DataLoader for the validation dataset, or None if no validation dataset is provided.
    """
    mean, std = mean_std_map[args.dataset.upper()]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.input_size,
                scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )

    train_ds, val_ds = get_dataset_from_name(args, transform=train_transform)

    train_loader = get_loader(train_ds, args, is_train=True)
    if val_ds is not None:
        val_loader = get_loader(val_ds, args, is_train=False)
    else:
        val_loader = None

    return train_loader, val_loader


def get_loader(ds, args, is_train=True):
    if args.multiprocessing_distributed:
        num_tasks = misc.get_world_size()
        args.global_rank = misc.get_rank()
        sampler = torch.utils.data.DistributedSampler(
            ds, num_replicas=num_tasks, rank=args.global_rank, shuffle=is_train
        )
        print(f"Train sampler for {args.global_rank} = {sampler}")
    else:
        sampler = torch.utils.data.RandomSampler(ds)

    loader = torch.utils.data.DataLoader(
        ds,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=True,
    )
    return loader


def get_dataset_from_name(args, **kwargs):
    dataset = args.dataset
    transform = kwargs.get("transform", None)

    if dataset.upper() == "LLC4320_SST_GLOBAL":
        _args = vars(args)
        _cls = SSTDatasetSingleNumpyTile
        try:
            train_dir = get_sst_data_dir(args, is_train=True)
        except ValueError:
            train_dir = args.data_path
        train_ds = _cls(
            train_dir,
            transform=transform,
            sea_ice_files=_args.get("sea_ice_data_path", None),
        )
        if _args.get("evaluate_mae_predictions", False):
            try:
                val_dir = get_sst_data_dir(args, is_train=False)
            except ValueError:
                val_dir = args.val_data_path
            test_ds = _cls(
                val_dir,
                transform=transform,
                sea_ice_files=_args.get("sea_ice_val_data_path", None),
            )
        else:
            test_ds = None
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
    return train_ds, test_ds
