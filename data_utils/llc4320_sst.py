# define a pytorch dataset for netCDF files, where each sample is a single 128x128 tile
import os
from glob import glob
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset

Tmin_jinbo = 8.292981147766113
Tmax_jinbo = 35.004295349121094
Tmean_jinbo = 16.739186747170315
Tstd_jinbo = 2.0404984487599913

Tmean_llc_with_ice = 12.81035
Tstd_llc_with_ice = 11.039073

Tmean_llc = 14.849469265947322  # 2012, without under-sea ice
Tstd_llc = 9.38270549534829


def get_sst_data_dir(args, is_train):
    """Handles whether provided args.data_path has train/val subfolders, and returns the correct one based on is_train.
    Args:
        args (Namespace): MAE arguments
        is_train (bool): Whether to return the train split. Only used if there are "train" and "val" subfolders in args.data_path.

    Returns:
        str: Path to a folder containing netCDF files. (Can also be a Zarr store)
    """

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    train_val_dirs_exist = os.path.isdir(train_dir) and os.path.isdir(val_dir)
    # If we have train & val subfolders, set data_dir to either of those depending on `is_train`
    if train_val_dirs_exist:
        data_dir = train_dir if is_train else val_dir
    elif (  # Handle val_data_path argument
        hasattr(args, "val_data_path")
        and os.path.isdir(args.val_data_path)
        and not is_train
    ):
        data_dir = args.val_data_path
    elif not train_val_dirs_exist and not is_train:
        raise ValueError(
            "No validation data found. Please provide a path to validation data using the `--val_data_path` argument."
        )
    else:  # Otherwise, just use args.data_path
        data_dir = args.data_path
    return data_dir


class SSTDatasetSingleNumpyTile(VisionDataset):
    """
    Torchvision-compatible dataset using sea surface temperature values stored in individual npy tiles.
    For the SingleTiles dataset, each npy file is assumed to only contain ONE SST field. In other words,
    each netCDF file contains a single 2D array of shape 64x64, 128x128, or 256x256, etc.

    Extends the base torchvision.datasets.vision.VisionDataset class.

    Args:
        root (string): Root directory of dataset, consisting of multiple npy files (*.npy).
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        sea_ice_files: Optional[Callable] = None,
    ) -> None:
        super(SSTDatasetSingleNumpyTile, self).__init__(
            root, transforms, transform, target_transform
        )
        if not root.endswith(".npy"):
            self.npy_files = glob(os.path.join(root, "*.npy*"))
            if sea_ice_files:
                under_sea_tiles = open(sea_ice_files, "r").read().splitlines()
                npy_filenames = list(map(os.path.basename, self.npy_files))
                npy_files_no_ice = sorted(set(npy_filenames) - set(under_sea_tiles))
                self.npy_files = [os.path.join(root, f) for f in npy_files_no_ice]
        else:
            self.npy_files = [root]

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        sst = self.get_sst(idx)

        if self.transform:
            sst = self.transform(Image.fromarray(sst))
        else:
            sst = torch.from_numpy(np.expand_dims(sst, 0))
        return sst, 0  # always return the 0 class

    def get_sst(self, idx):
        try:
            return np.load(self.npy_files[idx]).astype(np.float32)
        except:
            print(f"failed to load {self.npy_files[idx]}")
            return np.load(self.npy_files[0]).astype(np.float32)
