import numpy as np
import os
import sys
import random
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
import sklearn
import torch.nn.functional as F
from torch.utils.data import Subset
from pathlib import Path
from utils.dataset_utils import check_alt
import math

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sklearn.utils.check_random_state(seed)

num_clients = 4
runs = 1
dir_path = "brain_seg/"

class BraTSSegmentationDataset(Dataset):

    def __init__(self, root_dir, modality, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.split = split
        self.transform = transform
        self.series_paths = []

        assert modality in ["T1", "T1Gd", "T2", "FLAIR"], \
            f"Invalid BraTS modality: {modality}"

        # Map modality → filename pattern
        modality_pattern = {
            "T1": "*_t1.nii.gz",
            "T1Gd": "*_t1Gd.nii.gz",
            "T2": "*_t2.nii.gz",
            "FLAIR": "*_flair.nii.gz",
        }[modality]

        # Iterate over BraTS subsets (GBM/LGG naming varies by release)
        for subset in ["GBM_split"]:
            subset_dir = self.root_dir / subset / split
            if not subset_dir.exists():
                continue

            for subject_dir in sorted(subset_dir.iterdir()):
                if not subject_dir.is_dir():
                    continue

                # Image volume for selected modality
                image_files = list(subject_dir.glob(modality_pattern))

                # Ground-truth segmentation (BraTS)
                mask_files = list(
                    subject_dir.glob("*_GlistrBoost_ManuallyCorrected.nii.gz")
                )

                if image_files and mask_files:
                    self.series_paths.append(
                        (image_files[0], mask_files[0])
                    )

    def __len__(self):
        return len(self.series_paths)

    def __getitem__(self, idx):
        image_path, mask_path = self.series_paths[idx]

        if self.transform is not None:
            return self.transform(str(image_path), str(mask_path), self.modality)

        return str(image_path), str(mask_path), self.modality


from collections import defaultdict
from typing import Dict, List, Tuple

def distribute_samples_among_clients(datasets):
    client_data = {}
    global_ds_idx = 0
    client_id = 0

    print("=== Data Distribution Log ===")

    for modality, ds_list in datasets.items():
        for ds in ds_list:
            num_samples = len(ds)
            indices = list(range(num_samples))
            mid = num_samples // 2

            split_1 = indices[:mid]
            split_2 = indices[mid:]

            client_data[client_id] = [(global_ds_idx, split_1)]
            client_data[client_id + 1] = [(global_ds_idx, split_2)]

            print(
                f"Dataset {global_ds_idx} ({modality}) | "
                f"Total samples: {num_samples}"
            )
            print(
                f"  -> Client {client_id}:   {len(split_1)} samples "
                f"(idx {split_1[0]} to {split_1[-1]})"
            )
            print(
                f"  -> Client {client_id+1}: {len(split_2)} samples "
                f"(idx {split_2[0]} to {split_2[-1]})"
            )

            client_id += 2
            global_ds_idx += 1

    print("=== Distribution Complete ===\n")
    return client_data

def generate_brain_seg(id):

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train" + id + "/"
    val_path = dir_path + "val" + id + "/"
    test_path = dir_path + "test" + id + "/"

    if check_alt(config_path, train_path, val_path, test_path, num_clients, 1):
        return

    # Dataset parameters
    Brats_root_dir = './BraTS/'
    # BraTS modalities
    brats_modalities = ["T2", "FLAIR"]

    datasets = {}

    for modality in brats_modalities:
        # Train split
        brats_train = BraTSSegmentationDataset(
            Brats_root_dir,
            modality=modality,
            split="train"
        )

        # Validation split
        brats_val = BraTSSegmentationDataset(
            Brats_root_dir,
            modality=modality,
            split="val"
        )

        # Combine train + val
        brats_dataset = ConcatDataset([brats_train, brats_val])

        # Store per-modality dataset
        datasets[modality] = [brats_dataset]

    dataset_list = [ds for ds_list in datasets.values() for ds in ds_list]

    total_len = 0
    for dataset in dataset_list:
        print(len(dataset))
        total_len += len(dataset)
    print(f"Total samples: {total_len}")

    # Distribute to 4 clients
    client_data = distribute_samples_among_clients(
        datasets=datasets,
    )

    all_test_imgs = []
    all_test_masks = []
    all_test_mods = []

    # First pass: create train/val/test splits and collect test data
    for client_id, dataset_info in client_data.items():
        img_paths = []
        mask_paths = []
        modalities = []

        for dset_id, indices in dataset_info:
            dataset = dataset_list[dset_id]
            for idx in indices:
                img_path, mask_path, modality = dataset[idx]
                img_paths.append(img_path)
                mask_paths.append(mask_path)
                modalities.append(modality)

        # Shuffle
        combined = list(zip(img_paths, mask_paths, modalities))
        random.shuffle(combined)
        img_paths, mask_paths, modalities = zip(*combined)

        # Split
        train_split = int(0.8 * len(img_paths))
        val_split = int((len(img_paths) - train_split) / 2)

        train_imgs = img_paths[:train_split]
        train_masks = mask_paths[:train_split]
        train_mods = modalities[:train_split]

        val_imgs = img_paths[train_split:train_split + val_split]
        val_masks = mask_paths[train_split:train_split + val_split]
        val_mods = modalities[train_split:train_split + val_split]

        test_imgs = img_paths[train_split + val_split:]
        test_masks = mask_paths[train_split + val_split:]
        test_mods = modalities[train_split + val_split:]

        # Save train and val normally
        np.savez(os.path.join(train_path, f"{client_id}.npz"),
                 data={"x": train_imgs, "y": train_masks, "z": train_mods})
        np.savez(os.path.join(val_path, f"{client_id}.npz"),
                 data={"x": val_imgs, "y": val_masks, "z": val_mods})
        np.savez(os.path.join(test_path, f"{client_id}.npz"),
                 data={"x": test_imgs, "y": test_masks, "z": test_mods})

if __name__ == "__main__":
    for i in range(runs):
        set_seed(i)
        generate_brain_seg(str(i))
