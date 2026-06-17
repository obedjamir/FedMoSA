import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from flcore.clients.clientbase import Client
import matplotlib.pyplot as plt
import time

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1.):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class clientMSA(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        self.args = args
        self.current_step = 0
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_scaling = args.lam_bce
        self.dice_scaling = args.lam_dice
        self.proto_scaling = args.lam_proto

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )

    def train(self, tune=False):
        trainloader = self.load_train_data()
        print("Loaded Data!")

        self.model.train()

        train_steps = self.local_steps

        for epoch_idx in range(train_steps):
            epoch_loss = 0.0
            num_batches = 0
            self.current_step += 1

            for i, (x, y, z) in enumerate(trainloader):
                image_mask_pairs = self.load_images(x, y)
                images, masks = zip(*image_mask_pairs)

                x = torch.stack(list(images)).to(self.device)
                y = torch.stack(list(masks)).to(self.device)
                y = (y >= 0.5).float()

                self.optimizer.zero_grad()
                output = self.model(x)

                bce_loss = self.bce(output, y)
                dice_loss = self.dice(output, y)
                main_loss = self.bce_scaling * bce_loss + self.dice_scaling * dice_loss
                total_loss = main_loss

                total_loss.backward()

                self.optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1
                #break
