import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict

import json
import os
from typing import List, Tuple, Dict, Callable
from collections import defaultdict
from batchgenerators.utilities.file_and_folder_operations import save_json, join

# -------------------------------------------------------------------
# 1) Adapter definition (unchanged)
# -------------------------------------------------------------------
class Adapter(nn.Module):
    def __init__(self, hidden_size, reduction_factor=16, adapter_scaling=False):
        super().__init__()
        reduced_dim = hidden_size // reduction_factor
        self.down = nn.Linear(hidden_size, reduced_dim)
        self.relu = nn.ReLU()
        self.up = nn.Linear(reduced_dim, hidden_size)
        self.scaling = None

        if adapter_scaling:
            self.scaling = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if self.scaling is not None:
            return self.up(self.relu(self.down(x))) * self.scaling
        else:
            return self.up(self.relu(self.down(x)))


# -------------------------------------------------------------------
# 2) Wrapper block with no_grad on frozen parts + optional checkpoint
# -------------------------------------------------------------------
class MedicalAdapterBlock(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        hidden_size: int,
        adapter_reduction: int = 16,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.original_block = original_block
        self.adapter_attn = Adapter(hidden_size, adapter_reduction)
        self.adapter_mlp  = Adapter(hidden_size, adapter_reduction, adapter_scaling=True)
        self.use_checkpoint = use_checkpoint

        # freeze all original params
        for p in self.original_block.parameters():
            p.requires_grad = False

    def forward(self, x):
        def _forward(x):
            residual = x
            x_norm = self.original_block.norm1(x)
            with torch.no_grad():
                attn_out = self.original_block.attn(x_norm)
            x = residual + attn_out + self.adapter_attn(x_norm)

            residual = x
            x_norm = self.original_block.norm2(x)
            with torch.no_grad():
                mlp_out = self.original_block.mlp(x_norm)
            x = residual + mlp_out + self.adapter_mlp(x_norm)
            return x

        return checkpoint(_forward, x) if self.use_checkpoint else _forward(x)

# -------------------------------------------------------------------
# 3) Full SAM + adapters module
# -------------------------------------------------------------------
class MSAVanilla(nn.Module):
    def __init__(
        self,
        checkpoint_path: str = "./sam_vit_b_01ec64.pth",
        adapter_reduction: int = 16,
        use_checkpoint: bool = False
    ):
        super().__init__()
        # load base SAM
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self._inject_adapters(
            self.sam,
            adapter_reduction,
            use_checkpoint
        )

        # freeze *everything* except adapters
        for name, p in self.sam.image_encoder.named_parameters():
            if "adapter_attn" not in name and "adapter_mlp" not in name:
                p.requires_grad = False

    def _inject_adapters(
        self,
        sam_model,
        adapter_reduction,
        use_checkpoint
    ):
        blocks = sam_model.image_encoder.blocks
        hidden_size = blocks[0].mlp.lin1.in_features
        device = blocks[0].mlp.lin1.weight.device

        for i, orig_blk in enumerate(blocks):
            wrapped = MedicalAdapterBlock(
                original_block=orig_blk,
                hidden_size=hidden_size,
                adapter_reduction=adapter_reduction,
                use_checkpoint=use_checkpoint
            ).to(device)
            blocks[i] = wrapped


    def forward(self, images: torch.Tensor):
        image_embeddings = self.sam.image_encoder(images)
        pos_enc = self.sam.prompt_encoder.get_dense_pe()
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None
        )
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=pos_enc,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        return low_res_masks



class SAMVanilla(nn.Module):
    def __init__(self, checkpoint_path="./sam_vit_b_01ec64.pth"):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)

    def forward(self, images):
        image_embeddings = self.sam.image_encoder(images)

        pos_enc = self.sam.prompt_encoder.get_dense_pe()
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=pos_enc,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        return low_res_masks

class ClientFeatureCollector:
    def __init__(self, max_points=1000, stride=1):
        # data[key][client_id] -> list of tensors
        self.data = defaultdict(lambda: defaultdict(list))
        self.current_client = None
        self.max_points = max_points
        self.stride = stride

    def set_client(self, client_id):
        """Set active client (e.g., client index or name)."""
        self.current_client = client_id

    def _num_points(self, key, client_id):
        if key not in self.data or client_id not in self.data[key]:
            return 0
        return sum(t.shape[0] for t in self.data[key][client_id])

    def add(self, key, tensor):
        """
        key: identifier for adapter/layer (e.g., 'adapter_1')
        tensor: feature tensor of shape [B, T, C] or similar
        """
        if self.current_client is None:
            return

        # Token subsampling (memory-safe)
        feats = tensor.detach().cpu()[:, ::self.stride]
        feats = feats.reshape(-1, feats.shape[-1])  # flatten tokens → points

        current_pts = self._num_points(key, self.current_client)

        # Optional cap (kept commented to match your original)
        # if current_pts >= self.max_points:
        #     return
        #
        # remaining = self.max_points - current_pts
        # if feats.shape[0] > remaining:
        #     feats = feats[:remaining]

        self.data[key][self.current_client].append(feats)

    def get(self, key, client_id):
        if key not in self.data or client_id not in self.data[key]:
            raise KeyError(f"No data for key='{key}', client='{client_id}'")
        return torch.cat(self.data[key][client_id], dim=0)

    def clear(self):
        self.data.clear()
        self.current_client = None

class AdapterFeatureCollector:
    def __init__(self, max_points=1000, stride=1):
        self.data = defaultdict(lambda: defaultdict(list))
        self.current_modality = None
        self.max_points = max_points
        self.stride = stride

    def set_modality(self, modality):
        self.current_modality = modality

    def _num_points(self, key, modality):
        if key not in self.data or modality not in self.data[key]:
            return 0
        return sum(t.shape[0] for t in self.data[key][modality])

    def add(self, key, tensor):
        if self.current_modality is None:
            return

        # token subsample (safer on memory)
        feats = tensor.detach().cpu()[:, ::self.stride]
        feats = feats.reshape(-1, feats.shape[-1])  # flatten tokens → points

        current_pts = self._num_points(key, self.current_modality)
        #if current_pts >= self.max_points:
        #    return

        #remaining = self.max_points - current_pts
        #if feats.shape[0] > remaining:
        #    feats = feats[:remaining]

        self.data[key][self.current_modality].append(feats)

    def get(self, key, modality):
        if key not in self.data or modality not in self.data[key]:
            raise KeyError(f"No data for key='{key}', modality='{modality}'")
        return torch.cat(self.data[key][modality], dim=0)

    def clear(self):
        self.data.clear()
        self.current_modality = None



# ================================================================
# PrototypeAccumulator
# ------------------------------------------------
# - Maintains BOTH:
#   (1) accumulated prototype = mean over batches
#   (2) current-batch prototype = last forward only
# - No gradients, purely statistical
# ================================================================
class PrototypeAccumulator(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.register_buffer("sum", torch.zeros(feat_dim))
        self.register_buffer("count", torch.zeros(1))
        self.last = None

    def update(self, feats):
        batch_proto = feats.mean(dim=0)

        self.last = batch_proto

        self.sum += batch_proto.detach()
        self.count += 1

    def get(self):
        if self.count.item() == 0:
            return None
        return self.sum / self.count

    def get_current(self):
        return self.last

    def flush(self):
        self.sum.zero_()
        self.count.zero_()
        self.last = None

# ================================================================
# BottleneckAdapter
# ------------------------------------------------
# - Standard bottleneck adapter
# - Owns a PrototypeAccumulator
# - Can optionally collect prototypes during forward
# ================================================================
class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size, reduction_factor):
        super().__init__()
        reduced_dim = hidden_size // reduction_factor
        self.down = nn.Linear(hidden_size, reduced_dim)
        self.relu = nn.ReLU()
        self.up = nn.Linear(reduced_dim, hidden_size)
        self.prototypes = PrototypeAccumulator(hidden_size)

    def forward(self, x, collect_proto=False):
        out = self.up(self.relu(self.down(x)))
        if collect_proto:
            self.prototypes.update(out)
        return out

    def get_prototype(self):
        return self.prototypes.get()

    def get_current_prototype(self):
        return self.prototypes.get_current()

    def flush_prototype(self):
        self.prototypes.flush()


# ================================================================
# ModalityConditionalAdapter (Dynamic, BraTS-ready)
# ------------------------------------------------
# - Hard-routed adapters
# - One expert per modality (defined at init)
# - Modality list passed explicitly
# - No prototype collection
# ================================================================
class ModalityConditionalAdapter(nn.Module):
    def __init__(self, hidden_size, modalities, reduction_factor=16):
        super().__init__()

        assert isinstance(modalities, (list, tuple)) and len(modalities) > 0, \
            "modalities must be a non-empty list"

        self.modalities = list(modalities)

        # One expert per modality
        self.experts = nn.ModuleList([
            BottleneckAdapter(hidden_size, reduction_factor)
            for _ in self.modalities
        ])

        # Modality → expert index
        self.modality_to_expert = {
            m: i for i, m in enumerate(self.modalities)
        }

        # Default modality (safe)
        self.current_modality = self.modalities[0]

    def set_modality(self, modality: str):
        if modality not in self.modality_to_expert:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Expected one of {self.modalities}"
            )
        self.current_modality = modality

    def forward(self, x):
        B, H, W, D = x.shape
        x_flat = x.view(B * H * W, D)

        idx = self.modality_to_expert[self.current_modality]
        out = self.experts[idx](x_flat, collect_proto=False)

        return out.view(B, H, W, D)
# ================================================================
# SharedAdapter
# ------------------------------------------------
# - Cross-modality shared adapter
# - ONLY component that collects prototypes
# ================================================================
class SharedAdapter(nn.Module):
    def __init__(self, hidden_size, reduction_factor):
        super().__init__()
        self.adapter = BottleneckAdapter(hidden_size, reduction_factor)

    def forward(self, x, collect_proto=False):
        B, H, W, D = x.shape
        out = self.adapter(x.view(B * H * W, D), collect_proto)
        return out.view(B, H, W, D)

    def get_prototype(self):
        return self.adapter.get_prototype()

    def get_current_prototype(self):
        return self.adapter.get_current_prototype()

    def flush_prototype(self):
        self.adapter.flush_prototype()


# ================================================================
# MedicalAdapterBlockMC
# ------------------------------------------------
# - SAM transformer block with:
#   * modality-specific adapters (attn + mlp)
#   * shared adapters (attn + mlp)
# - Original SAM block is frozen
# ================================================================
class MedicalAdapterBlockMC(nn.Module):
    def __init__(
        self,
        original_block,
        hidden_size,
        modalities,
        adapter_reduction=16,
        use_checkpoint=False,
    ):
        super().__init__()
        self.original_block = original_block

        self.adapter_attn = ModalityConditionalAdapter(
            hidden_size, modalities, adapter_reduction
        )
        self.adapter_mlp = ModalityConditionalAdapter(
            hidden_size, modalities, adapter_reduction
        )

        self.shared_adapter_attn = SharedAdapter(hidden_size, adapter_reduction)
        self.shared_adapter_mlp = SharedAdapter(hidden_size, adapter_reduction)

        self.use_checkpoint = use_checkpoint

        for p in self.original_block.parameters():
            p.requires_grad = False

    def forward(self, x):
        collect_proto = getattr(self, "_collect_proto", False)

        def _forward(x):
            residual = x
            x_norm = self.original_block.norm1(x)
            with torch.no_grad():
                attn_out = self.original_block.attn(x_norm)

            x = (
                residual
                + attn_out
                + self.adapter_attn(x_norm)
                + self.shared_adapter_attn(x_norm, collect_proto)
            )

            residual = x
            x_norm = self.original_block.norm2(x)
            with torch.no_grad():
                mlp_out = self.original_block.mlp(x_norm)

            x = (
                residual
                + mlp_out
                + self.adapter_mlp(x_norm)
                + self.shared_adapter_mlp(x_norm, collect_proto)
            )
            return x

        return checkpoint(_forward, x) if self.use_checkpoint else _forward(x)

    def get_shared_prototypes(self):
        return {
            "attn_shared": self.shared_adapter_attn.get_prototype(),
            "mlp_shared": self.shared_adapter_mlp.get_prototype(),
        }

    def get_current_shared_prototypes(self):
        return {
            "attn_shared": self.shared_adapter_attn.get_current_prototype(),
            "mlp_shared": self.shared_adapter_mlp.get_current_prototype(),
        }

    def flush_shared_prototypes(self):
        self.shared_adapter_attn.flush_prototype()
        self.shared_adapter_mlp.flush_prototype()

# ================================================================
# SAMModalityAdapter
# ------------------------------------------------
# - Full SAM model with injected adapter blocks
# - Supports:
#   * modality switching
#   * prototype collection
#   * block-wise prototype access
# ================================================================
class MoSA(nn.Module):
    def __init__(
        self,
        checkpoint_path="./sam_vit_b_01ec64.pth",
        modalities=["MR", "CT"],
        adapter_reduction=16,
        use_checkpoint=False,
    ):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self._inject_adapters(
            self.sam, adapter_reduction, use_checkpoint, modalities
        )

        for name, p in self.sam.image_encoder.named_parameters():
            if not any(k in name for k in [
                "adapter_attn",
                "adapter_mlp",
                "shared_adapter_attn",
                "shared_adapter_mlp",
            ]):
                p.requires_grad = False

    def _inject_adapters(self, sam_model, adapter_reduction, use_checkpoint, modalities):
        blocks = sam_model.image_encoder.blocks
        hidden_size = blocks[0].mlp.lin1.in_features
        device = blocks[0].mlp.lin1.weight.device

        for i, blk in enumerate(blocks):
            blocks[i] = MedicalAdapterBlockMC(
                blk,
                hidden_size,
                modalities,
                adapter_reduction,
                use_checkpoint,
            ).to(device)

    def set_modality(self, modality: str):
        for blk in self.sam.image_encoder.blocks:
            blk.adapter_attn.set_modality(modality)
            blk.adapter_mlp.set_modality(modality)

    def forward(self, images, collect_proto=True):
        for blk in self.sam.image_encoder.blocks:
            blk._collect_proto = collect_proto

        image_embeddings = self.sam.image_encoder(images)

        pos_enc = self.sam.prompt_encoder.get_dense_pe()
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=pos_enc,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return low_res_masks

    def get_all_shared_prototypes(self):
        return [blk.get_shared_prototypes() for blk in self.sam.image_encoder.blocks]

    def get_all_current_shared_prototypes(self):
        return [
            blk.get_current_shared_prototypes()
            for blk in self.sam.image_encoder.blocks
        ]

    def flush_all_shared_prototypes(self):
        for blk in self.sam.image_encoder.blocks:
            blk.flush_shared_prototypes()











def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def compute_pooling(shape: List[int], min_feature_map_size: int = 8):
    """
    nnU-Net v2–style dynamic pooling:
    Pool per axis independently until feature map becomes too small.
    """
    shape = list(shape)
    pool_per_axis = [0] * len(shape)

    while True:
        can_pool = [s >= 2 * min_feature_map_size for s in shape]
        if not any(can_pool):
            break

        for i in range(len(shape)):
            if can_pool[i]:
                shape[i] //= 2
                pool_per_axis[i] += 1

    num_pool = max(pool_per_axis)
    return num_pool, pool_per_axis


class NNUNetV2Planner:
    """
    Slice-only (2D) nnU-Net v2–style planner.

    Assumptions:
    - 2D slices only
    - Pre-normalized intensities
    - Single modality
    - Binary segmentation
    - No physical spacing metadata
    """

    def __init__(self, dataset_name: str, output_folder: str):
        self.dataset_name = dataset_name
        self.output_folder = output_folder

    # ----------------------------------------------------------
    # Fingerprint Collection (nnU-Net v2 style)
    # ----------------------------------------------------------
    def collect_dataset_statistics(
        self,
        trainloader,
        load_images_fn,
        num_samples: int = None
    ) -> Dict:

        print("🔍 Collecting dataset statistics (slice-based nnU-Net v2)...")

        shapes = []
        spacings = []
        foreground_intensities = []
        num_samples_processed = 0

        for batch_idx, (x, y, _) in enumerate(trainloader):

            if num_samples and num_samples_processed >= num_samples:
                break

            image_mask_pairs = load_images_fn(x, y)
            images, masks = zip(*image_mask_pairs)

            for img, mask in zip(images, masks):

                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()

                if img.ndim == 3:
                    img = img.squeeze()
                if mask.ndim == 3:
                    mask = mask.squeeze()

                mask_binary = (mask > 0)

                if mask_binary.sum() > 0:
                    fg_pixels = img[mask_binary]
                    foreground_intensities.extend(fg_pixels.flatten().tolist())

                shapes.append(img.shape)
                spacings.append([1.0, 1.0])  # Slice assumption
                num_samples_processed += 1

        shapes = np.array(shapes)
        spacings = np.array(spacings)

        foreground_intensities = (
            np.array(foreground_intensities)
            if len(foreground_intensities) > 0
            else np.array([0.0])
        )

        fingerprint = {
            "spacing": np.median(spacings, axis=0).tolist(),
            "shape_after_cropping": np.median(shapes, axis=0).tolist(),
            "raw_shapes": shapes.tolist(),
            "raw_spacings": spacings.tolist(),
            "median_image_size_in_voxels": np.median(shapes, axis=0).tolist(),
            "foreground_intensity_properties_per_channel": {
                "0": {
                    "mean": float(np.mean(foreground_intensities)),
                    "median": float(np.median(foreground_intensities)),
                    "std": float(np.std(foreground_intensities)),
                    "min": float(np.min(foreground_intensities)),
                    "max": float(np.max(foreground_intensities)),
                    "percentile_00_5": float(np.percentile(foreground_intensities, 0.5)),
                    "percentile_99_5": float(np.percentile(foreground_intensities, 99.5)),
                }
            },
            "num_channels": 1,
            "num_samples": num_samples_processed,
        }

        print(f"✅ Collected statistics from {num_samples_processed} slices")
        print(f"   Median shape: {fingerprint['shape_after_cropping']}")

        return fingerprint

    # ----------------------------------------------------------
    # dataset.json (nnU-Net v2 compliant)
    # ----------------------------------------------------------
    def create_dataset_json(self, fingerprint: Dict):
    
        channel_names = {"0": "channel_0"}
    
        dataset_json = {
            "name": self.dataset_name,
            "description": "Binary 2D slice dataset",
            "labels": {
                "background": 0,
                "foreground": 1,
            },
            "numTraining": fingerprint["num_samples"],
            "channel_names": channel_names,
            "file_ending": ".npy",
        }
    
        save_path = join(self.output_folder, "dataset.json")
        save_json(dataset_json, save_path, sort_keys=False)
    
        print("✅ Saved dataset.json")
        return dataset_json

    # ----------------------------------------------------------
    # nnUNetPlans.json (v2 architecture format)
    # ----------------------------------------------------------
    def create_plans(self, fingerprint: Dict):
    
        median_shape = fingerprint["shape_after_cropping"]
        median_spacing = fingerprint["spacing"]
    
        # Patch size
        patch_size = [
            int(np.clip(median_shape[0], 128, 512)),
            int(np.clip(median_shape[1], 128, 512)),
        ]
    
        # Dynamic pooling
        num_pool, pool_per_axis = compute_pooling(patch_size)
        divisor = 2 ** num_pool
        patch_size = [(p // divisor) * divisor for p in patch_size]
    
        n_stages = num_pool + 1
        base_features = 32
        max_features = 512
    
        features_per_stage = [
            min(base_features * (2 ** i), max_features)
            for i in range(n_stages)
        ]
    
        strides = [[1, 1]] + [[2, 2]] * num_pool
        kernel_sizes = [[3, 3]] * n_stages
    
        arch_kwargs = {
            "n_stages": n_stages,
            "features_per_stage": features_per_stage,
            "conv_op": "torch.nn.modules.conv.Conv2d",
            "kernel_sizes": kernel_sizes,
            "strides": strides,
            "n_conv_per_stage": [2] * n_stages,
            "n_conv_per_stage_decoder": [2] * (n_stages - 1),
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
            "norm_op_kwargs": {
                "eps": 1e-5,
                "affine": True,
            },
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {
                "negative_slope": 0.01,
                "inplace": True,
            },
        }
    
        configuration_2d = {
            "data_identifier": "nnUNetPlans_2d",
            "preprocessor_name": "DefaultPreprocessor",
            "batch_size": 12,
            "patch_size": patch_size,
            "spacing": median_spacing,
            "normalization_schemes": ["NoNormalization"],
            "use_mask_for_norm": [False],
        
            "UNet_class_name": "PlainConvUNet",
        
            "UNet_base_num_features": base_features,
            "unet_max_num_features": max_features,
        
            "conv_kernel_sizes": kernel_sizes,
            "pool_op_kernel_sizes": strides,
        
            "n_conv_per_stage_encoder": [2] * n_stages,
            "n_conv_per_stage_decoder": [2] * (n_stages - 1),
        
            "architecture": {
                "network_class_name": "PlainConvUNet",
                "arch_kwargs": arch_kwargs,
                "_kw_requires_import": [
                    "conv_op",
                    "norm_op",
                    "dropout_op",
                    "nonlin",
                ],
            },
        }
    
        plans = {
            "dataset_name": self.dataset_name,
            "plans_name": "nnUNetPlans",
            "original_median_spacing_after_transp": median_spacing,
            "original_median_shape_after_transp": median_shape,
            "image_reader_writer": "NumpyIO",
            "transpose_forward": [0, 1],
            "transpose_backward": [0, 1],
            "configurations": {"2d": configuration_2d},
            "experiment_planner_used": "SliceBasedExperimentPlanner",
            "label_manager": "LabelManager",
        }
    
        save_path = join(self.output_folder, "nnUNetPlans.json")
        save_json(plans, save_path, sort_keys=False)
    
        print("✅ Generated nnUNetPlans.json (slice-based faithful v2)")
        print(f"   Patch size: {patch_size}")
        print(f"   Pooling stages: {num_pool}")

        return plans
