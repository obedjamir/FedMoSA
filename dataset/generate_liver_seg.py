import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import sklearn
import torch.nn.functional as F
from torch.utils.data import Subset
from pathlib import Path
from utils.dataset_utils import check_alt

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sklearn.utils.check_random_state(seed)

num_clients = 5
runs = 5
dir_path = "liver_seg/"

# Dataset Class
class LiTSDataset(Dataset):
    def __init__(self, root_dir, train=True):
        """
        Args:
            root_dir (str): Path to the root directory of the dataset.
            train (bool): If True, uses training data; otherwise, uses test data.
        """
        if train:
            ct_dir = os.path.join(root_dir, "train_CT")
            mask_dir = os.path.join(root_dir, "train_mask")
        else:
            ct_dir = os.path.join(root_dir, "test_CT")
            mask_dir = os.path.join(root_dir, "test_mask")

        # Get sorted paths to CT volumes and mask files
        self.ct_paths = sorted(Path(ct_dir).glob('volume-*.nii'))
        self.mask_paths = sorted(Path(mask_dir).glob('segmentation-*.nii'))
        
        assert len(self.ct_paths) == len(self.mask_paths), "CT and Mask counts mismatch"

    def __len__(self):
        """Returns the number of studies (volumes)."""
        return len(self.ct_paths)

    def __getitem__(self, idx):
        """
        Returns the paths to the full CT volume and corresponding mask.
        Args:
            idx (int): Index of the volume.
        Returns:
            tuple: (ct_volume_path, mask_volume_path)
        """
        ct_path = self.ct_paths[idx]
        mask_path = self.mask_paths[idx]

        return str(ct_path), str(mask_path), "CT"

# Valid series IDs from user input
VALID_SERIES_IDS = {
    '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
    '16', '17', '18', '19', '20', '21', '22', '23', '25', '26', '27', '29',
    '30', '31', '32', '34', '36', '38', '43', '60', '503', '504', '603',
    '604', '705', '706', '803', '804', '904', '2205', '2206'
}

class DLDSSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory of the dataset.
            transform (callable, optional): Optional transform to be applied later.
        """
        self.root_dir = Path(root_dir)
        self.patient_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        self.series_paths = []

        # Collect all valid series (image/mask folder pairs)
        for patient_dir in self.patient_dirs:
            series_dirs = [d for d in patient_dir.iterdir() if d.is_dir() and d.name in VALID_SERIES_IDS]

            for series_dir in series_dirs:
                img_dir = series_dir / "images"
                mask_dir = series_dir / "masks"

                # Ensure both image and mask directories exist
                if img_dir.exists() and mask_dir.exists():
                    self.series_paths.append((img_dir, mask_dir))

    def __len__(self):
        """Returns the number of valid series (image/mask folder pairs)."""
        return len(self.series_paths)

    def __getitem__(self, idx):
        """
        Returns the paths to the images and masks directory for a series.
        Args:

            idx (int): Index of the series.
        Returns:
            tuple: (images_folder_path, masks_folder_path)
        """
        img_dir, mask_dir = self.series_paths[idx]
        return str(img_dir), str(mask_dir), "MR"

class CHAOSSegmentationDataset(Dataset):
    def __init__(self, root_dir, modality, train=True):
        """
        Args:
            root_dir (str): Path to the CHAOS dataset (e.g., "/DATA1/CHAOS").
            modality (str): Modality type, either "CT" or "MR".
        """
        if train:
            root_dir = os.path.join(root_dir, "Train_Sets")
        else:
            root_dir = os.path.join(root_dir, "Test_Sets")

        root_dir = Path(root_dir) / modality
        patient_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
        self.series_paths = []
        self.modality = modality

        if modality=="CT":
            for patient_dir in patient_dirs:
                img_dir = patient_dir / "DICOM_anon"
                mask_dir = patient_dir / "Ground"

                if img_dir.exists() and mask_dir.exists():
                    self.series_paths.append((img_dir, mask_dir))

        elif modality=="MR":
            for patient_dir in patient_dirs:
                img_dir = patient_dir / "T1DUAL" / "DICOM_anon" / "InPhase"
                mask_dir = patient_dir / "T1DUAL" / "Ground"

                if img_dir.exists() and mask_dir.exists():
                    self.series_paths.append((img_dir, mask_dir))

            for patient_dir in patient_dirs:
                img_dir = patient_dir / "T1DUAL" / "DICOM_anon" / "OutPhase"
                mask_dir = patient_dir / "T1DUAL" / "Ground"

                if img_dir.exists() and mask_dir.exists():
                    self.series_paths.append((img_dir, mask_dir))

            for patient_dir in patient_dirs:
                img_dir = patient_dir / "T2SPIR" / "DICOM_anon"
                mask_dir = patient_dir / "T2SPIR" / "Ground"

                if img_dir.exists() and mask_dir.exists():
                    self.series_paths.append((img_dir, mask_dir))

    def __len__(self):
        """Returns the number of valid patient directories."""
        return len(self.series_paths)

    def __getitem__(self, idx):
        """
        Returns the paths to the images and masks directory for a patient.
        Args:
            idx (int): Index of the patient.
        Returns:
            tuple: (images_folder_path, masks_folder_path)
        """
        img_dir, mask_dir = self.series_paths[idx]
        return str(img_dir), str(mask_dir), self.modality

class IRCADLiverDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Path to the root directory containing 3Dircadb patient folders.
        """
        self.patient_dirs = sorted(Path(root_dir).glob("3Dircadb*"))
        self.valid_pairs = []

        for patient_dir in self.patient_dirs:
            ct_dir = patient_dir / "PATIENT_DICOM"
            mask_dir = patient_dir / "MASKS_DICOM" / "liver"

            if ct_dir.is_dir() and mask_dir.is_dir():
                self.valid_pairs.append((ct_dir, mask_dir))

        if not self.valid_pairs:
            raise RuntimeError("No valid CT/mask pairs found.")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (ct_dicom_dir, liver_mask_dicom_dir, modality)
        """
        ct_dir, mask_dir = self.valid_pairs[idx]
        return str(ct_dir), str(mask_dir), "CT"

class CTORGDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Path to the CT-ORG dataset directory containing volume-*.nii.gz and labels-*.nii.gz files.
        """
        self.root_dir = Path(root_dir)

        self.ct_paths = sorted(self.root_dir.glob("volume-*.nii.gz"))
        self.mask_paths = sorted(self.root_dir.glob("labels-*.nii.gz"))

        assert len(self.ct_paths) == len(self.mask_paths), "Mismatch in number of volumes and masks"

        # Optional: verify IDs match (e.g., volume-123 vs labels-123)
        for ct, mask in zip(self.ct_paths, self.mask_paths):
            ct_id = ct.stem.split('-')[-1]
            mask_id = mask.stem.split('-')[-1]
            assert ct_id == mask_id, f"ID mismatch: {ct.name} vs {mask.name}"

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, idx):
        ct_path = self.ct_paths[idx]
        mask_path = self.mask_paths[idx]
        return str(ct_path), str(mask_path), "CT"

from collections import defaultdict
from typing import Dict, List, Tuple

def distribute_samples_among_clients(
    datasets: Dict[str, List],
    template: Dict[str, int],   # template decides total clients
    min_per_modality: int,
    max_per_ds: int
) -> Dict[int, List[Tuple[int, List[int]]]]:
    
    # 1. Preprocess datasets into flat list with modality tracking
    all_datasets = []
    modality_per_ds = []
    dataset_map = {}
    for modality, ds_list in datasets.items():
        for ds_idx, ds in enumerate(ds_list):
            global_idx = len(all_datasets)
            all_datasets.append(ds)
            modality_per_ds.append(modality)
            dataset_map[global_idx] = (modality, ds_idx, ds)

    # 2. Build client modality assignments from template
    modality_assignments = {}
    client_id = 0
    for combo, n_clients in template.items():
        combo_modalities = combo.split("+")  # "MR+CT" → ["MR","CT"]
        for _ in range(n_clients):
            modality_assignments[client_id] = combo_modalities
            client_id += 1
    num_clients = client_id  # total number of clients comes from template

    # 3. Build modality sample pools
    modality_pools = defaultdict(list)
    for global_idx, ds in enumerate(all_datasets):
        modality = modality_per_ds[global_idx]
        for sample_idx in range(len(ds)):
            modality_pools[modality].append((global_idx, sample_idx))
    for modality in modality_pools:
        random.shuffle(modality_pools[modality])

    # 4. Init structures
    client_data = {c: defaultdict(list) for c in range(num_clients)}
    modality_counts = {c: defaultdict(int) for c in range(num_clients)}
    ds_counts = {c: defaultdict(int) for c in range(num_clients)}

    # 5. First pass: give each client the minimum per modality
    for client_id in range(num_clients):
        for modality in modality_assignments[client_id]:
            needed = min_per_modality
            pool = modality_pools[modality].copy()
            while needed > 0 and pool:
                ds_idx, sample_idx = pool.pop()
                if ds_counts[client_id][ds_idx] < max_per_ds:
                    client_data[client_id][ds_idx].append(sample_idx)
                    ds_counts[client_id][ds_idx] += 1
                    modality_counts[client_id][modality] += 1
                    needed -= 1
                else:
                    modality_pools[modality].insert(0, (ds_idx, sample_idx))
            modality_pools[modality] = pool
            if needed > 0:
                raise RuntimeError(f"Client {client_id} couldn't get {min_per_modality} samples for {modality}")

    # 6. Second pass: distribute leftovers
    for modality, pool in modality_pools.items():
        if not pool:
            continue
        clients = [c for c in range(num_clients) if modality in modality_assignments[c]]
        if not clients:
            continue
        samples_by_ds = defaultdict(list)
        for (ds_idx, sample_idx) in pool:
            samples_by_ds[ds_idx].append(sample_idx)
        for ds_idx, samples in samples_by_ds.items():
            eligible_clients = [c for c in clients if ds_counts[c][ds_idx] < max_per_ds]
            if eligible_clients:
                idx = 0
                for sample in samples:
                    client_id = eligible_clients[idx % len(eligible_clients)]
                    client_data[client_id][ds_idx].append(sample)
                    ds_counts[client_id][ds_idx] += 1
                    modality_counts[client_id][modality] += 1
                    idx += 1
            else:
                min_count = min(ds_counts[c][ds_idx] for c in clients)
                min_clients = [c for c in clients if ds_counts[c][ds_idx] == min_count]
                idx = 0
                for sample in samples:
                    client_id = min_clients[idx % len(min_clients)]
                    client_data[client_id][ds_idx].append(sample)
                    ds_counts[client_id][ds_idx] += 1
                    modality_counts[client_id][modality] += 1
                    idx += 1

    # 7. Summary print
    print("\n=== Client Data Distribution Summary ===")
    total_mr_count, total_ct_count = 0, 0
    modalities = list(datasets.keys())
    for client_id in range(num_clients):
        total_samples = 0
        modality_summary = []
        ct_count, mr_count = 0, 0
        for ds_idx, samples in client_data[client_id].items():
            total_samples += len(samples)
            modality = modality_per_ds[ds_idx]
            for sample_idx in samples:
                _, _, tag = all_datasets[ds_idx][sample_idx]
                if tag == 'CT':
                    ct_count += 1
                elif tag == 'MR':
                    mr_count += 1
        for modality in modalities:
            count = modality_counts[client_id][modality]
            modality_summary.append(f"{modality}: {count}")
        total_mr_count += mr_count
        total_ct_count += ct_count
        print(f"Client {client_id}: {total_samples} samples "
              f"({', '.join(modality_summary)}) | {ct_count+mr_count}")
    print(f"MR Total: {total_mr_count}")
    print(f"CT Total: {total_ct_count}")
    print(f"Total: {total_mr_count+total_ct_count}")

    return {c: [(ds_idx, samples) for ds_idx, samples in client_data[c].items()] for c in range(num_clients)}

def generate_liver_seg(template, id):
    num_clients = 0
    for combo, n_clients in template.items():
        for _ in range(n_clients):
            num_clients += 1

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train" + id + "/"
    val_path = dir_path + "val" + id + "/"
    test_path = dir_path + "test" + id + "/"

    if check_alt(config_path, train_path, val_path, test_path, num_clients, 1):
        return

    # Dataset parameters
    DLDS_root_dir = './DLDS/Segmentation'
    LiTS17_root_dir = './LiTS17/data'
    CHAOS_root_dir = './CHAOS'
    IRCADb_root_dir = './IRCADb'
    CTORG_root_dir = './CTORG/CT-ORG/OrganSegmentations'

    # Load datasets
    dlds_dataset = DLDSSegmentationDataset(DLDS_root_dir)
    total = len(dlds_dataset)
    mid = total // 2
    indices = list(range(total))
    dlds_dataset1 = Subset(dlds_dataset, indices[:mid])
    dlds_dataset2 = Subset(dlds_dataset, indices[mid:])

    lits_train = LiTSDataset(LiTS17_root_dir, train=True)
    lits_test = LiTSDataset(LiTS17_root_dir, train=False)
    lits_dataset = ConcatDataset([lits_train, lits_test])

    chaos_ct_dataset = CHAOSSegmentationDataset(CHAOS_root_dir, "CT", train=True)

    chaos_mr_dataset = CHAOSSegmentationDataset(CHAOS_root_dir, "MR", train=True)

    ircadb_ct_dataset = IRCADLiverDataset(IRCADb_root_dir)
    ctorg_ct_dataset = CTORGDataset(CTORG_root_dir)

    datasets = {
        "MR": [dlds_dataset1, dlds_dataset2, chaos_mr_dataset],
        "CT": [lits_dataset, chaos_ct_dataset, ircadb_ct_dataset, ctorg_ct_dataset]
    }

    dataset_list = [dlds_dataset1, dlds_dataset2, chaos_mr_dataset, lits_dataset, chaos_ct_dataset, ircadb_ct_dataset, ctorg_ct_dataset]

    total_len = 0
    for dataset in dataset_list:
        print(len(dataset))
        total_len += len(dataset)
    print(f"Total samples: {total_len}")
    # Distribute to 5 clients
    client_data = distribute_samples_among_clients(
        datasets=datasets,
        template=template,
        min_per_modality=40,
        max_per_ds=200
    )

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

        test_imgs = img_paths[train_split:train_split + val_split]
        test_masks = mask_paths[train_split:train_split + val_split]
        test_mods = modalities[train_split:train_split + val_split]

        val_imgs = img_paths[train_split + val_split:]
        val_masks = mask_paths[train_split + val_split:]
        val_mods = modalities[train_split + val_split:]

        # Save train and val normally
        np.savez(os.path.join(train_path, f"{client_id}.npz"),
                 data={"x": train_imgs, "y": train_masks, "z": train_mods})

        np.savez(os.path.join(test_path, f"{client_id}.npz"),
                 data={"x": test_imgs, "y": test_masks, "z": test_mods})

        np.savez(os.path.join(val_path, f"{client_id}.npz"),
                 data={"x": val_imgs, "y": val_masks, "z": val_mods})

templates = [
{
    "MR+CT": 1,
    "MR": 2,
    "CT": 2
},

{
    "MR+CT": 2,
    "MR": 1,
    "CT": 2
},

{
    "MR+CT": 2,
    "MR": 2,
    "CT": 1
},

{
    "MR+CT": 5,
    "MR": 0,
    "CT": 0
},

{
    "MR+CT": 0,
    "MR": 3,
    "CT": 2
}
]

if __name__ == "__main__":
    for i in range(runs):
        set_seed(i)
        generate_liver_seg(templates[i], str(i))
