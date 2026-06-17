import copy
import torch
import numpy as np
import os
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Dataset
from concurrent.futures import ThreadPoolExecutor
from utils.data_utils import read_client_data
from ..trainmodel.models import *
import torchvision.transforms as transforms
import torch
import heapq
from collections import OrderedDict
from PIL import Image

def _get_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def save_mask_tensor_as_images(tensor, image_paths, out_dir="./images", suffix="mask"):
    os.makedirs(out_dir, exist_ok=True)
    tensor = tensor.detach().cpu()

    if tensor.dim() == 4 and tensor.size(1) == 1:
        tensor = tensor.squeeze(1)

    for i in range(tensor.size(0)):
        mask_np = tensor[i].numpy()
        mask_img = (mask_np * 255).astype(np.uint8)

        fname = _get_stem(image_paths[i])
        save_path = os.path.join(out_dir, f"{fname}_{suffix}.png")

        Image.fromarray(mask_img).save(save_path)


def save_tensor_images(tensor, image_paths, out_dir="./images", suffix="image"):
    os.makedirs(out_dir, exist_ok=True)
    tensor = tensor.detach().cpu()

    if tensor.dim() == 4 and tensor.size(1) == 3:
        tensor = tensor[:, 0:1, :, :]

    if tensor.dim() == 4 and tensor.size(1) == 1:
        tensor = tensor.squeeze(1)

    for i in range(tensor.size(0)):
        img_np = tensor[i].numpy()
        img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)

        fname = _get_stem(image_paths[i])
        save_path = os.path.join(out_dir, f"{fname}_{suffix}.png")

        Image.fromarray(img_uint8).save(save_path)

class TopIoUOverlayCollector:
    def __init__(
        self,
        out_dir="./images",
        top_k=30,
        out_size=(512, 512),
        pred_color=(0, 114, 178),   # red
        gt_color=(0, 255, 0),     # green
        alpha=0.4
    ):
        self.out_dir = out_dir
        self.top_k = top_k
        self.out_size = out_size
        self.pred_color = pred_color
        self.gt_color = gt_color
        self.alpha = alpha

        # dataset -> min-heap of (iou, counter, entry)
        self.buffers = {}
        self.counter = 0

        os.makedirs(out_dir, exist_ok=True)

    @staticmethod
    def compute_iou(pred, gt):
        pred = pred > 0.5
        gt = gt > 0.5
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        return inter / (union + 1e-8)

    @staticmethod
    def undo_normalize(img):
        # undo Normalize(mean=0.5, std=0.5)
        img = img * 0.5 + 0.5
        return (img * 255).clip(0, 255).astype(np.uint8)

    def add_batch(self, image_tensor, y_pred, y_true, image_paths):
        image_tensor = image_tensor.detach().cpu()
        y_pred = y_pred.detach().cpu()
        y_true = y_true.detach().cpu()

        # force single-channel image
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[:, 0, :, :]

        if y_pred.dim() == 4:
            y_pred = y_pred.squeeze(1)
        if y_true.dim() == 4:
            y_true = y_true.squeeze(1)

        for i in range(image_tensor.size(0)):
            fname = os.path.basename(image_paths[i])
            dataset = fname[:4]

            img = image_tensor[i].numpy()
            pred = y_pred[i].numpy()
            gt = y_true[i].numpy()

            iou = self.compute_iou(pred, gt)

            entry = {
                "img": img,
                "pred": pred,
                "gt": gt,
                "fname": fname,
                "iou": iou,
            }

            if dataset not in self.buffers:
                self.buffers[dataset] = []

            heap = self.buffers[dataset]

            item = (iou, self.counter, entry)
            self.counter += 1

            if len(heap) < self.top_k:
                heapq.heappush(heap, item)
            else:
                if iou > heap[0][0]:
                    heapq.heapreplace(heap, item)

    def _overlay_and_save(self, img, mask, color, save_path):
        # image
        img = self.undo_normalize(img)
        img = Image.fromarray(img).resize(self.out_size, Image.BILINEAR)
        img = np.array(img)

        overlay = np.stack([img, img, img], axis=-1)

        # mask
        mask = Image.fromarray(mask).resize(self.out_size, Image.NEAREST)
        mask = np.array(mask) > 0.5

        for c in range(3):
            overlay[..., c][mask] = (
                (1 - self.alpha) * overlay[..., c][mask]
                + self.alpha * color[c]
            ).astype(np.uint8)

        Image.fromarray(overlay).save(save_path)

    def save_all(self):
        for dataset, heap in self.buffers.items():
            dataset_dir = os.path.join(self.out_dir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)

            # sort by descending IoU
            heap_sorted = sorted(heap, key=lambda x: -x[0])

            for rank, (iou, _, entry) in enumerate(heap_sorted):
                base = os.path.splitext(entry["fname"])[0]
                tag = f"iou{float(iou):.3f}_rank{rank+1}"

                # prediction overlay
                self._overlay_and_save(
                    entry["img"],
                    entry["pred"],
                    self.pred_color,
                    os.path.join(dataset_dir, f"{base}_{tag}_pred.png")
                )

                # ground-truth overlay
                self._overlay_and_save(
                    entry["img"],
                    entry["gt"],
                    self.gt_color,
                    os.path.join(dataset_dir, f"{base}_{tag}_gt.png")
                )

class MergedDataset(Dataset):
    def __init__(self, datasets_by_modality, batch_size):
        self.batch_size = batch_size

        # Store datasets
        self.datasets = OrderedDict(datasets_by_modality)

        # Trim datasets to full batches
        self.batched_data = {}
        self.num_batches = {}

        for modality, ds in self.datasets.items():
            n_batches = len(ds) // batch_size
            self.num_batches[modality] = n_batches
            self.batched_data[modality] = ds[: n_batches * batch_size]

        # Total batches
        self.total_batches = sum(self.num_batches.values())

        # Precompute ordering with ORIGINAL rule
        self.mixed_indices = self._create_strict_alternating_indices()

    def _create_strict_alternating_indices(self):
        # Sort modalities by number of batches (descending)
        modalities_sorted = sorted(
            self.num_batches.keys(),
            key=lambda m: self.num_batches[m],
            reverse=True
        )

        # First modality = one with most batches
        first = modalities_sorted[0]
        rest = modalities_sorted[1:]

        # Batch pointers
        batch_ptr = {m: 0 for m in self.num_batches}
        mixed = []

        # Strict alternation while all modalities still have batches
        while True:
            added = False

            # First modality first
            if batch_ptr[first] < self.num_batches[first]:
                mixed.append((first, batch_ptr[first]))
                batch_ptr[first] += 1
                added = True

            # Then all other modalities (in fixed order)
            for m in rest:
                if batch_ptr[m] < self.num_batches[m]:
                    mixed.append((m, batch_ptr[m]))
                    batch_ptr[m] += 1
                    added = True

            if not added:
                break

        return mixed

    def __len__(self):
        return len(self.mixed_indices)

    def __getitem__(self, idx):
        modality, batch_idx = self.mixed_indices[idx]

        start = batch_idx * self.batch_size
        end = (batch_idx + 1) * self.batch_size
        data_batch = self.batched_data[modality][start:end]

        img_paths, mask_paths, modalities = zip(*data_batch)
        return list(img_paths), list(mask_paths), list(modalities)

class Client(object):

    def __init__(self, args, id, dataset_id, shared_model, **kwargs):
        self.args = args
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.dataset_id = dataset_id
        self.labels = None
        self.num_classes = None
        self.current_round = None

        self.mod_sample_count = None
        self.train_samples = None
        self.modality_count = None
        self.model = None
        self.global_shared_prototypes = None

        if shared_model is not None:
            self.model = copy.deepcopy(shared_model)

        self.train_data = None
        self.test_data = None
        self.val_data = None

        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.collector=None

        self.transform_x_sam = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transform_y_sam = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transform_x_nnunet = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transform_y_nnunet = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.set_seed(32)

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    def compute_prototype_alignment_loss(self):
        if self.global_shared_prototypes is None:
            return None

        batch_protos = self.model.get_all_current_shared_prototypes()

        proto_loss = 0.0
        count = 0

        for blk_idx, blk_proto in enumerate(batch_protos):
            global_blk_proto = self.global_shared_prototypes[blk_idx]

            for k in ["attn_shared", "mlp_shared"]:
                lp = blk_proto[k]
                gp = global_blk_proto[k]

                if lp is None or gp is None:
                    continue

                proto_loss += torch.norm(lp - gp, p=2)
                count += 1

        if count == 0:
            return None

        return proto_loss / count
        torch.backends.cudnn.deterministic = True

    def load_train_data(self, batch_size=None, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size

        self.train_data = read_client_data(self.dataset, self.current_round, self.id, self.dataset_id, data_split='train') if self.train_data is None else self.train_data
        self.train_samples = len(self.train_data)

        if self.args.model_name == "mosa":
            modality_data = {}
            for item in self.train_data:
                modality_data.setdefault(item[2], []).append(item)

            self.mod_sample_count = {m: 0 for m in self.args.mod_list}
            for item in self.train_data:
                m = item[2]
                if m in modality_data:
                    modality_data[m].append(item)
                    self.mod_sample_count[m] += 1

            if shuffle:
                for m in modality_data:
                    random.shuffle(modality_data[m])

            for m in modality_data:
                num_batches = len(modality_data[m]) // batch_size
                modality_data[m] = modality_data[m][: num_batches * batch_size]

            print(f"Client {self.id} train data → " + ", ".join(f"{m}: {len(v)}" for m, v in modality_data.items()))

            merged_dataset = MergedDataset(modality_data, batch_size=batch_size)
            return DataLoader(merged_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])

        batch_size = min(batch_size, len(self.train_data))
        return DataLoader(self.train_data, batch_size, drop_last=True, shuffle=shuffle)


    def load_test_data(self, batch_size=None, ood=False):
        if batch_size is None:
            batch_size = self.batch_size

        self.test_data = read_client_data(self.dataset, self.current_round, self.id, self.dataset_id, data_split='test') if self.test_data is None else self.test_data

        print("Loading Test Data...")

        modality_data = {}
        for item in self.test_data:
            modality_data.setdefault(item[2], []).append(item)

        for m in modality_data:
            random.shuffle(modality_data[m])

        for m in modality_data:
            num_batches = len(modality_data[m]) // batch_size
            modality_data[m] = modality_data[m][: num_batches * batch_size]

        print(f"Client {self.id} test data → " + ", ".join(f"{m}: {len(v)}" for m, v in modality_data.items()))

        merged_dataset = MergedDataset(modality_data, batch_size=batch_size)
        return DataLoader(merged_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])

    def load_val_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        self.val_data = read_client_data(self.dataset, self.current_round, self.id, self.dataset_id, data_split='val') if self.val_data is None else self.val_data

        if self.args.model_name == "mosa":
            modality_data = {}
            for item in self.val_data:
                modality_data.setdefault(item[2], []).append(item)

            for m in modality_data:
                random.shuffle(modality_data[m])

            for m in modality_data:
                num_batches = len(modality_data[m]) // batch_size
                modality_data[m] = modality_data[m][: num_batches * batch_size]

            print(f"Client {self.id} val data → " + ", ".join(f"{m}: {len(v)}" for m, v in modality_data.items()))

            merged_dataset = MergedDataset(modality_data, batch_size=batch_size)
            return DataLoader(merged_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])

        batch_size = min(batch_size, len(self.val_data))
        return DataLoader(self.val_data, batch_size, drop_last=False, shuffle=False)

    def test_metrics(self, temp_model=None, val=True, ood = False):
        if val:
            testloaderfull = self.load_val_data()
        else:
            testloaderfull = self.load_test_data(ood=ood)

        model = self.get_eval_model(temp_model)
        model.eval()

        test_loss = 0.0
        iou_scores = []
        dice_scores = []
        num_batches = 0
        #img_collector = TopIoUOverlayCollector(out_dir="msa_images")

        with torch.no_grad():
            for i, (x, y, z) in enumerate(testloaderfull):
                image_mask_pairs = self.load_images(x, y)
                images, masks = zip(*image_mask_pairs)

                x = torch.stack(list(images)).to(self.device)
                y = torch.stack(list(masks)).to(self.device)
                y_true = (y > 0.5).float()

                if self.args.model_name == "mosa":
                    modality = z[0]
                    model.set_modality(modality)

                output = model(x)
                # Main segmentation loss
                bce_loss = self.bce(output, y_true)
                dice_loss = self.dice(output, y_true)
                main_loss = self.bce_scaling * bce_loss + self.dice_scaling * dice_loss
                total_loss = main_loss

                proto_loss = self.compute_prototype_alignment_loss()
                if proto_loss is not None:
                    total_loss += 0.1 * proto_loss

                test_loss += total_loss.item()

                # Predictions
                y_prob = torch.sigmoid(output)
                y_pred = (y_prob > 0.5).float()
                #img_collector.add_batch(image_tensor=x,y_pred=y_pred,y_true=y_true,image_paths=img_pths)

                # IoU
                intersection = torch.sum(y_pred * y_true)
                union = torch.sum(y_pred) + torch.sum(y_true) - intersection
                iou = intersection / torch.clamp(union, min=1.0)
                iou_scores.append(iou.item())

                # Dice
                dice = (2.0 * intersection) / torch.clamp(torch.sum(y_pred) + torch.sum(y_true), min=1.0)
                dice_scores.append(dice.item())
                num_batches += 1
                #break

        #img_collector.save_all()
        mean_iou = sum(iou_scores) / len(iou_scores)
        mean_dice = sum(dice_scores) / len(dice_scores)
        mean_test_loss = test_loss / num_batches
        return mean_iou, mean_dice, mean_test_loss, num_batches * self.batch_size

    def set_parameters(self, model):
        src_params = dict(model.named_parameters())
        tgt_params = dict(self.model.named_parameters())

        for name in tgt_params:
            if name in src_params:
                if tgt_params[name].shape == src_params[name].shape:
                    tgt_params[name].data.copy_(src_params[name].data)

    def normalize_img(self, img):
        mean = img.mean()
        std = img.std() + 1e-8
        return ((img - mean) / std).astype(np.float32)

    def get_eval_model(self, temp_model=None):
        model = self.model_per if hasattr(self, "model_per") else self.model
        return model

    def load_image(self, image_name, mask_name, rotate=True):
        with Image.open(image_name) as image, Image.open(mask_name) as mask:
            x_image = image.convert('L')
            y_mask = mask.convert('L')

            if rotate:
                angle = random.choice([90, 180, 270])
                x_image = x_image.rotate(angle)
                y_mask = y_mask.rotate(angle)

            if self.args.model_name in ["mosa", "msavanilla"]:
                x_image = self.transform_x_sam(x_image)
                y_mask = self.transform_y_sam(y_mask)
                x_image = x_image.repeat(3, 1, 1)
            else:
                x_image = self.transform_x_nnunet(x_image)
                y_mask = self.transform_y_nnunet(y_mask)

        return x_image, y_mask

    def load_images(self, image_names, mask_names, rotate=True, max_workers=32):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pairs = list(executor.map(self.load_image, image_names, mask_names, [rotate] * len(image_names)))
        return pairs
