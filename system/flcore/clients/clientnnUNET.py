import torch
import torch.nn as nn
from flcore.clients.clientbase import Client
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from ..trainmodel.models import *
import random
import pickle
random.seed(32)

# -------------------------
# One-hot conversion
# -------------------------
def convert_to_onehot(binary_mask, num_classes=2):
    return torch.cat([1 - binary_mask, binary_mask], dim=1)


# -------------------------
# Client definition
# -------------------------
class clientnnUNET(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)

        self.args = args
        self.device = args.device
        self.current_step = 0
        self.patch_size = None

        # nnU-Net v2 loss (binary)
        self.loss_fn = DC_and_BCE_loss(
            soft_dice_kwargs={"smooth": 1e-5, "do_bg": False},
            bce_kwargs={}
        )

        # Initialize planner
        self.planner = NNUNetV2Planner(
            dataset_name=f"{self.args.dataset}_{self.args.times}",
            output_folder=f"nnUNet_{self.id}"
        )

        # Collect fingerprint
        trainloader = self.load_train_data()
        self.fingerprint = self.planner.collect_dataset_statistics(
            trainloader=trainloader,
            load_images_fn=self.load_images,
            num_samples=self.train_samples
        )
        if self.args.afa==1:
            self.initialize_model(self.fingerprint)

    # --------------------------------------------------
    # Model initialization
    # --------------------------------------------------
    def initialize_model(self, fingerprint):

        dataset_json = self.planner.create_dataset_json(fingerprint)
        plans = self.planner.create_plans(fingerprint)

        plans_manager = PlansManager(plans)

        config_name = "2d"
        configuration_manager = plans_manager.get_configuration(config_name)

        config = configuration_manager.configuration
        arch = config["architecture"]

        num_input_channels = len(dataset_json["channel_names"])
        num_output_channels = len(dataset_json["labels"])

        self.model = get_network_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=True
        ).to(self.args.device)

        # Official nnU-Net v2 optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.99,
            weight_decay=3e-5,
            nesterov=False
        )
        self.patch_size = configuration_manager.configuration["patch_size"]

        print("✅ nnU-Net v2 slice-based model initialized")
        print(f"   Input channels: {num_input_channels}")
        print(f"   Output channels: {num_output_channels}")
        print(f"   Stages: {arch['arch_kwargs']['n_stages']}")

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    def train(self):

        trainloader = self.load_train_data()
        self.model.train()

        for epoch in range(self.local_steps):

            epoch_loss = 0.0
            num_batches = 0

            for x, y, _ in trainloader:

                image_mask_pairs = self.load_images(x, y)
                images, masks = zip(*image_mask_pairs)

                patch_images = []
                patch_masks = []

                for img, msk in zip(images, masks):
                    img_patch, msk_patch = self.sample_patch(img, msk)
                    patch_images.append(img_patch)
                    patch_masks.append(msk_patch)

                images = torch.stack(patch_images).to(self.device)
                masks = torch.stack(patch_masks).to(self.device)


                targets = (masks > 0.5).float()
                targets_onehot = convert_to_onehot(targets)

                self.optimizer.zero_grad()

                outputs = self.model(images)

                # ---------------------------
                # Deep supervision handling
                # ---------------------------
                if isinstance(outputs, (list, tuple)):

                    # nnU-Net v2 style weights
                    weights = [1 / (2 ** i) for i in range(len(outputs))]
                    weight_sum = sum(weights)
                    weights = [w / weight_sum for w in weights]

                    loss = 0.0

                    for output, w in zip(outputs, weights):

                        # Resize target if resolution mismatch
                        if output.shape[2:] != targets_onehot.shape[2:]:
                            resized_target = F.interpolate(
                                targets_onehot,
                                size=output.shape[2:],
                                mode="nearest"
                            )
                        else:
                            resized_target = targets_onehot

                        loss += w * self.loss_fn(output, resized_target)

                else:
                    # No deep supervision case
                    loss = self.loss_fn(outputs, targets_onehot)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                #break

    def test_metrics(self, temp_model=None, val=True, ood = False):

        if val:
            dataloader = self.load_val_data()
        else:
            dataloader = self.load_test_data(ood=ood)

        model = self.get_eval_model(temp_model)
        model.eval()

        total_loss = 0.0
        dice_scores = []
        iou_scores = []
        total_samples = 0

        with torch.no_grad():
            for x, y, _ in dataloader:

                image_mask_pairs = self.load_images(x, y)
                images, masks = zip(*image_mask_pairs)

                patch_images = []
                patch_masks = []

                for img, msk in zip(images, masks):
                    img_patch, msk_patch = self.sample_patch(img, msk)
                    patch_images.append(img_patch)
                    patch_masks.append(msk_patch)

                images = torch.stack(patch_images).to(self.device)
                masks = torch.stack(patch_masks).to(self.device)

                targets = (masks > 0.5).float()
                targets_onehot = torch.cat([1 - targets, targets], dim=1)

                outputs = model(images)

                # -------------------------
                # Loss (handle deep supervision)
                # -------------------------
                if isinstance(outputs, (list, tuple)):

                    weights = [1 / (2 ** i) for i in range(len(outputs))]
                    weight_sum = sum(weights)
                    weights = [w / weight_sum for w in weights]

                    loss = 0.0

                    for output, w in zip(outputs, weights):

                        if output.shape[2:] != targets_onehot.shape[2:]:
                            resized_target = F.interpolate(
                                targets_onehot,
                                size=output.shape[2:],
                                mode="nearest"
                            )
                        else:
                            resized_target = targets_onehot

                        loss += w * self.loss_fn(output, resized_target)

                    # Use full resolution output for metrics
                    main_output = outputs[0]

                else:
                    loss = self.loss_fn(outputs, targets_onehot)
                    main_output = outputs

                total_loss += loss.item()

                # -------------------------
                # Metrics (use full-res output only)
                # -------------------------
                logits_fg = main_output[:, 1]              # foreground channel
                probs_fg = torch.sigmoid(logits_fg)         # (B, H, W)

                preds = (probs_fg > 0.5).float()            # binary mask
                targets_fg = targets.squeeze(1)             # (B, H, W)

                intersection = (preds * targets_fg).sum()
                union = preds.sum() + targets_fg.sum() - intersection

                dice = (2 * intersection) / torch.clamp(
                    preds.sum() + targets_fg.sum(), min=1.0
                )
                iou = intersection / torch.clamp(union, min=1.0)
                dice_scores.append(dice.item())
                iou_scores.append(iou.item())

                total_samples += images.shape[0]
                #break

        mean_dice = sum(dice_scores) / len(dice_scores)
        mean_iou = sum(iou_scores) / len(iou_scores)
        mean_loss = total_loss / len(dice_scores)

        return mean_iou, mean_dice, mean_loss, total_samples


    def sample_patch(self, image, mask):
        """
        image: [C, H, W]
        mask:  [1, H, W]
        returns: cropped patch [C, Hp, Wp], [1, Hp, Wp]
        """

        _, H, W = image.shape
        Hp, Wp = self.patch_size

        # Pad if needed
        pad_h = max(0, Hp - H)
        pad_w = max(0, Wp - W)

        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h))
            mask = F.pad(mask, (0, pad_w, 0, pad_h))
            _, H, W = image.shape

        # Random crop coordinates
        y1 = random.randint(0, H - Hp)
        x1 = random.randint(0, W - Wp)

        image_patch = image[:, y1:y1+Hp, x1:x1+Wp]
        mask_patch = mask[:, y1:y1+Hp, x1:x1+Wp]

        return image_patch, mask_patch
