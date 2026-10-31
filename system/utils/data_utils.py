import numpy as np
import os
import nibabel as nib
import pydicom
import cv2
import random
import concurrent.futures
from multiprocessing import cpu_count
import pickle
import hashlib

def normalize_image(img, modality):
    percentile_map = {
        "CT": (0.5, 99.5),
        "MR": (2, 98),
        "T1": (1, 98),
        "T1Gd": (1, 98),
        "T2": (0.5, 99),
        "FLAIR": (0.5, 99),
    }

    p_low, p_high = percentile_map.get(modality, (1, 99))

    low_val = np.percentile(img, p_low)
    high_val = np.percentile(img, p_high)

    img = np.clip(img, low_val, high_val)
    img = (img - low_val) / (high_val - low_val + 1e-8)

    return (img * 255).astype(np.uint8)

def load_nii_slices(image_path, mask_path, modalities):
    image_data = nib.load(image_path).get_fdata(dtype=np.float32)
    mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)
    num_slices = image_data.shape[2]

    image_slices, mask_slices = [], []

    is_ct_org = ("CT-ORG" in image_path) or ("CT-ORG" in mask_path)

    for i in range(num_slices):
        img_slice = image_data[:, :, i]
        msk_slice = mask_data[:, :, i]

        if np.all(msk_slice == 0):
            continue

        if is_ct_org:
            if 1 not in msk_slice:
                continue
            msk_slice = (msk_slice == 1).astype(np.uint8) * 255

        modality = modalities[i] if isinstance(modalities, list) else modalities
        img_slice = normalize_image(img_slice, modality)

        image_slices.append(img_slice)
        mask_slices.append((msk_slice > 0).astype(np.uint8) * 255)

    return image_slices, mask_slices

def load_dicom_slices(image_dir, mask_dir, modalities):
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.dcm', '.dicom'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.dcm', '.dicom'))])

    images, masks = [], []

    for idx, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        mask_path = os.path.join(mask_dir, mask_file)
        msk = pydicom.dcmread(mask_path).pixel_array.astype(np.float32)

        unique_values = np.unique(msk)
        if len(unique_values) <= 1:
            continue

        msk = (msk == unique_values[1]).astype(np.uint8) * 255

        image_path = os.path.join(image_dir, image_file)
        img = pydicom.dcmread(image_path).pixel_array.astype(np.float32)

        modality = modalities[idx] if isinstance(modalities, list) else modalities
        img = normalize_image(img, modality)

        images.append(img)
        masks.append(msk)

    return images, masks

def load_png_slices(image_dir, mask_dir, modalities):
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    images, masks = [], []

    for idx, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        mask_path = os.path.join(mask_dir, mask_file)
        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        unique_values = np.unique(msk)
        if len(unique_values) <= 1:
            continue

        msk = (msk == unique_values[1]).astype(np.uint8) * 255

        image_path = os.path.join(image_dir, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        modality = modalities[idx] if isinstance(modalities, list) else modalities
        img = normalize_image(img, modality)

        images.append(img)
        masks.append(msk)

    return images, masks

def load_slices(image_path_or_dir, mask_path_or_dir, modalities):
    if os.path.isfile(image_path_or_dir) and image_path_or_dir.lower().endswith(('.nii', '.nii.gz')):
        return load_nii_slices(image_path_or_dir, mask_path_or_dir, modalities)

    if os.path.isdir(image_path_or_dir):
        image_files = os.listdir(image_path_or_dir)

        if any(f.lower().endswith(('.dcm', '.dicom')) for f in image_files):
            return load_dicom_slices(image_path_or_dir, mask_path_or_dir, modalities)

        return load_png_slices(image_path_or_dir, mask_path_or_dir, modalities)

    raise ValueError(f"Unsupported input type: {image_path_or_dir}")

TRAIN_FRACTION = 0.25        # share of every training volume's slices drawn afresh in each epoch


def train_seed(dataset, dataset_id, idx, round, epoch):
    """Deterministic seed of one client's training subset in one epoch of one round (same in every process)."""
    key = f"{dataset}|{dataset_id}|{idx}|{round}|{epoch}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def epoch_subset(volumes, seed, fraction=TRAIN_FRACTION):
    """A fresh random int(n * fraction) slices of every training volume -> [(image, mask, modality), ...]."""
    rng = random.Random(seed)
    out = []
    for modality, images, masks in volumes:
        for i in rng.sample(range(len(images)), int(len(images) * fraction)):
            out.append((images[i], masks[i], modality))
    return out


def subset_counts(volumes, fraction=TRAIN_FRACTION):
    """Slices per modality in every epoch's subset (the same in every epoch)."""
    counts = {}
    for modality, images, _ in volumes:
        counts[modality] = counts.get(modality, 0) + int(len(images) * fraction)
    return counts


def _load_png(path):
    """Exactly what the training loop used to read from a saved PNG slice (8-bit grayscale)."""
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert('L'))


def build_slices(dataset, idx, dataset_id, data_split):
    """Val / test: every slice of the client's volumes as (image uint8 HxW, mask uint8 HxW {0,255}, modality).
    The pixels are those of the PNG slices the previous pipeline wrote; an existing path cache (cache_*.pkl) is
    reused with its order."""
    data_dir = os.path.join('../dataset', dataset)
    old_cache = os.path.join(data_dir, f'cache_{idx}_{data_split}_{dataset_id}.pkl')

    if os.path.exists(old_cache):
        with open(old_cache, 'rb') as f:
            path_list = pickle.load(f)
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            imgs = list(executor.map(_load_png, [p[0] for p in path_list]))
            masks = list(executor.map(_load_png, [p[1] for p in path_list]))
        return [(im, mk, p[2]) for im, mk, p in zip(imgs, masks, path_list)]

    file_path = os.path.join(data_dir, f'{data_split}{dataset_id}/', f'{idx}.npz')
    with np.load(file_path, allow_pickle=True) as f:
        paths = f['data'].tolist()

    data_list = []
    for img_path, mask_path, modality in zip(paths['x'], paths['y'], paths['z']):
        img_slices, mask_slices = load_slices(img_path, mask_path, modality)
        for img_slice, mask_slice in zip(img_slices, mask_slices):
            data_list.append((np.ascontiguousarray(img_slice, dtype=np.uint8),
                              np.ascontiguousarray(mask_slice, dtype=np.uint8), modality))
    return data_list


def read_data(dataset, idx, dataset_id, data_split):
    """Val / test slices, built once into ../dataset/<dataset>/slices_<idx>_<split>_<dataset_id>.pkl."""
    slice_file = os.path.join('../dataset', dataset, f'slices_{idx}_{data_split}_{dataset_id}.pkl')
    if os.path.exists(slice_file):
        with open(slice_file, 'rb') as f:
            return pickle.load(f)
    data_list = build_slices(dataset, idx, dataset_id, data_split)
    with open(slice_file, 'wb') as f:
        pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data_list


def read_train_volumes(dataset, idx, dataset_id):
    """All slices of the client's training volumes, kept per volume: [(modality, [image], [mask]), ...] (uint8 HxW).
    Built once into ../dataset/<dataset>/slices_<idx>_train_<dataset_id>_volumes.pkl; every epoch draws its own
    subset with epoch_subset()."""
    data_dir = os.path.join('../dataset', dataset)
    vol_file = os.path.join(data_dir, f'slices_{idx}_train_{dataset_id}_volumes.pkl')
    if os.path.exists(vol_file):
        with open(vol_file, 'rb') as f:
            return pickle.load(f)

    with np.load(os.path.join(data_dir, f'train{dataset_id}/', f'{idx}.npz'), allow_pickle=True) as f:
        paths = f['data'].tolist()
    volumes = []
    for img_path, mask_path, modality in zip(paths['x'], paths['y'], paths['z']):
        img_slices, mask_slices = load_slices(img_path, mask_path, modality)
        volumes.append((modality,
                        [np.ascontiguousarray(m, dtype=np.uint8) for m in img_slices],
                        [np.ascontiguousarray(m, dtype=np.uint8) for m in mask_slices]))
    with open(vol_file, 'wb') as f:
        pickle.dump(volumes, f, protocol=pickle.HIGHEST_PROTOCOL)
    return volumes


def read_client_data(dataset, round, idx, dataset_id, data_split='train'):
    """train: the client's training volumes (see read_train_volumes); val / test: lists of slices."""
    if data_split == 'train':
        return read_train_volumes(dataset, idx, dataset_id)
    return read_data(dataset, idx, dataset_id, 'test' if data_split == 'test' else 'val')
