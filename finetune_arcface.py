# -*- coding: utf-8 -*-
"""
Optional fine-tuning: DINOv2-base + LoRA + ArcFace loss.

Use this only if zero-shot DINOv2 accuracy is insufficient for your domain
(e.g. photo-vs-scan gap, heavy foiling, non-standard crops).

After training, re-run build_faiss_index.py with USE_LORA_CHECKPOINT set
to True to rebuild the index with the fine-tuned embeddings.

Requirements (in addition to base requirements):
  peft>=0.10.0

Usage:
  python finetune_arcface.py
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
from time import time

from yugioh_utils import imgPathToRGBTensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_DIR        = './cardDatabaseFull/'
SAVE_PATH         = './savedModels/dino_lora_arcface.pth'
BATCH_SIZE        = 32
EPOCHS            = 30
LR                = 1e-4
LORA_RANK         = 16
LORA_ALPHA        = 32
ARCFACE_MARGIN    = 0.5
ARCFACE_SCALE     = 64
EMBEDDING_DIM     = 768

assert os.path.exists(TARGET_DIR)
os.makedirs('./savedModels', exist_ok=True)


# ---------------------------------------------------------------------------
# ArcFace loss
# ---------------------------------------------------------------------------
class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace).
    Converts metric learning into a classification problem — no triplet mining.
    Reference: https://arxiv.org/abs/1801.07698
    """

    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.scale   = scale
        self.margin  = margin
        self.weight  = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)   # threshold: cos(pi - m)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        # Normalise weights and embeddings
        cos_theta = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)

        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Numerical stability: fall back when cos(theta) < threshold
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m,
                                  cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        logits *= self.scale
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CardDataset(Dataset):
    """Returns an RGB tensor and class index for each card image."""

    def __init__(self, root):
        self.folder = dset.ImageFolder(root=root)
        self.imgs   = self.folder.imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        tensor = imgPathToRGBTensor(path).squeeze(0)   # (3, 224, 224)
        return tensor, label

    @property
    def num_classes(self):
        return len(self.folder.classes)


# ---------------------------------------------------------------------------
# Model: DINOv2 + LoRA adapter
# ---------------------------------------------------------------------------
def build_model(num_classes, device):
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model

    backbone = AutoModel.from_pretrained("facebook/dinov2-base")

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],   # attention projections in ViT
        lora_dropout=0.1,
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_cfg)
    backbone.print_trainable_parameters()

    arcface = ArcFaceLoss(EMBEDDING_DIM, num_classes,
                          margin=ARCFACE_MARGIN, scale=ARCFACE_SCALE)

    backbone = backbone.to(device)
    arcface  = arcface.to(device)
    return backbone, arcface


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def extract_cls(backbone, pixel_values):
    outputs = backbone(pixel_values=pixel_values)
    cls = outputs.last_hidden_state[:, 0, :]
    return F.normalize(cls, p=2, dim=1)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Training on {device}')

    dataset    = CardDataset(TARGET_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)

    backbone, arcface = build_model(dataset.num_classes, device)

    params    = list(backbone.parameters()) + list(arcface.parameters())
    optimizer = optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f'Training {EPOCHS} epochs, {len(dataset)} images, {dataset.num_classes} classes')

    for epoch in range(EPOCHS):
        backbone.train()
        epoch_loss = 0.0
        begin = time()

        for i, (imgs, labels) in enumerate(dataloader):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            embeddings = extract_cls(backbone, imgs)
            loss = arcface(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 20 == 0:
                print(f'  Epoch {epoch}  step {i}/{len(dataloader)}  loss {loss.item():.4f}')

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch}  avg_loss={avg_loss:.4f}  ({time()-begin:.1f}s)')

        # Save checkpoint each epoch
        torch.save({
            'epoch':        epoch,
            'backbone':     backbone.state_dict(),
            'arcface':      arcface.state_dict(),
            'num_classes':  dataset.num_classes,
        }, SAVE_PATH)

    print(f'Saved to {SAVE_PATH}')


if __name__ == '__main__':
    train()
