import numpy as np
import os
import nibabel as nib
import pydicom
import cv2
import imageio
import random
from torchvision import transforms
import concurrent.futures
from multiprocessing import cpu_count
import hashlib
import pickle

def encode_string_sha256(input_string):
    encoded = hashlib.sha256(input_string.encode()).hexdigest()
    return encoded

def apply_windowing(img, window_center=50, window_width=400):
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    img = np.clip(img, min_value, max_value)
    return (img - min_value) / (max_value - min_value + 1e-8) * 255

def process_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Lambda(lambda x: x.float()),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image)

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

def save_image_pair(img_slice, mask_slice, processed_dir, path_string, modality, vol_idx, slice_idx):
    sample_id = encode_string_sha256(path_string)
    img_id = modality + "_IMAGE_" + sample_id
    mask_id = modality + "_MASK_" + sample_id

    img_save_path = os.path.join(processed_dir, f"{img_id}_{vol_idx}_{slice_idx}.png")
    mask_save_path = os.path.join(processed_dir, f"{mask_id}_{vol_idx}_{slice_idx}.png")

    if not os.path.exists(img_save_path):
        imageio.imwrite(img_save_path, img_slice)
    if not os.path.exists(mask_save_path):
        imageio.imwrite(mask_save_path, mask_slice)
    return img_save_path, mask_save_path, modality

def get_seed(*args, cache_path="./seed_cache.pkl"):
    key = "_".join(map(str, args))

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            seed_cache = pickle.load(f)
    else:
        seed_cache = {}

    if key in seed_cache:
        #print(seed_cache[key])
        return seed_cache[key]

    seed_value = hash((args[0], args[1], args[2], args[3]))

    seed_cache[key] = seed_value
    with open(cache_path, "wb") as f:
        pickle.dump(seed_cache, f)

    return seed_value

def random_subset(dataset, round, idx, dataset_id, list1, list2, percentage=1.0):
    assert len(list1) == len(list2)

    # Seed for reproducibility across rounds
    random.seed(get_seed(dataset, round, idx, dataset_id))

    # Zip with indices to track original positions
    indexed_pairs = list(enumerate(zip(list1, list2)))
    subset_size = int(len(indexed_pairs) * percentage)

    # Random sampling with original indices
    sampled = random.sample(indexed_pairs, subset_size)
    indices, pairs = zip(*sampled) if sampled else ([], [])
    subset1, subset2 = zip(*pairs) if pairs else ([], [])
    return [list(subset1), list(subset2), list(indices)]

def read_data(dataset, round, idx, dataset_id, data_split='train'):
    version = 0.25
    data_dir = os.path.join('../dataset', dataset)
    file_path = os.path.join(data_dir, f'{data_split}{dataset_id}/', f'{idx}.npz')
    if data_split == "train":
        cache_file = os.path.join(data_dir, f'cache_{idx}_{round}_{data_split}_{dataset_id}_{version}.pkl')
    else:
        cache_file = os.path.join(data_dir, f'cache_{idx}_{data_split}_{dataset_id}.pkl')

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data_list = pickle.load(f)
            return data_list

    with np.load(file_path, allow_pickle=True) as f:
        paths = f['data'].tolist()

    data_list = []
    tasks = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        for vol_idx, (img_path, mask_path, modality) in enumerate(zip(paths['x'], paths['y'], paths['z'])):

            img_slices, mask_slices = load_slices(img_path, mask_path, modality)

            if data_split is not "train":
                for slice_idx, (img_slice, mask_slice) in enumerate(zip(img_slices, mask_slices)):
                    tasks.append(executor.submit(save_image_pair, img_slice, mask_slice, data_dir, img_path, modality, vol_idx, slice_idx))
            else:
                pair_subset = random_subset(dataset, round, idx, dataset_id, img_slices, mask_slices, percentage=version)
                subset_imgs, subset_masks, subset_indices = pair_subset

                for img_slice, mask_slice, slice_idx in zip(subset_imgs, subset_masks, subset_indices):
                    tasks.append(executor.submit(save_image_pair, img_slice, mask_slice, data_dir, img_path, modality, vol_idx, slice_idx))


        for future in concurrent.futures.as_completed(tasks):
            img_save_path, mask_save_path, slice_modality = future.result()
            data_list.append((img_save_path, mask_save_path, slice_modality))

    with open(cache_file, 'wb') as f:
        pickle.dump(data_list, f)

    return data_list

def read_client_data(dataset, round, idx, dataset_id, data_split='train'):
    if data_split == 'train':
        train_data = read_data(dataset, round, idx, dataset_id, 'train')
        return train_data
    elif data_split == 'test':
        test_data = read_data(dataset, round, idx, dataset_id, 'test')
        return test_data
    else:
        val_data = read_data(dataset, round, idx, dataset_id, 'val')
        return val_data
