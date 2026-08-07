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
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as T
from PIL import Image
from time import time

from yugioh_utils import imgArtCropper, _IMAGENET_MEAN, _IMAGENET_STD

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Set False to use clean images only (faster, lower robustness)
USE_ROBUST_AUGMENTATION = True

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
# Robust augmentation pipeline
# ---------------------------------------------------------------------------

class _GlareTransform:
    """Soft specular highlight — simulates foil/holographic cards."""
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, pil_img):
        if random.random() > self.p:
            return pil_img
        img = np.array(pil_img).astype(np.float32) / 255.0
        h, w = img.shape[:2]
        glare = np.zeros((h, w), dtype=np.float32)
        cx = random.randint(w // 5, 4 * w // 5)
        cy = random.randint(h // 5, 4 * h // 5)
        ax = random.randint(w // 8, w // 2)
        ay = random.randint(h // 8, h // 2)
        cv2.ellipse(glare, (cx, cy), (ax, ay), random.randint(0, 180), 0, 360, 1.0, -1)
        sigma = random.randint(10, 30)
        glare = cv2.GaussianBlur(glare, (sigma * 6 + 1, sigma * 6 + 1), sigma)
        glare *= random.uniform(0.25, 0.70)
        img = np.clip(img + glare[:, :, np.newaxis], 0.0, 1.0)
        return Image.fromarray((img * 255).astype(np.uint8))


class _MotionBlurTransform:
    """Directional blur — simulates camera shake or fast card movement."""
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, pil_img):
        if random.random() > self.p:
            return pil_img
        img = np.array(pil_img)
        ksize = random.choice([5, 7, 9, 11])
        K = np.zeros((ksize, ksize), dtype=np.float32)
        K[ksize // 2, :] = 1.0 / ksize
        M = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), random.uniform(0, 180), 1)
        K = cv2.warpAffine(K, M, (ksize, ksize))
        img = cv2.filter2D(img, -1, K)
        return Image.fromarray(img)


# Applied during training — teaches the model to be invariant to real-world
# corruptions so that a blurry/glary/occluded query still finds the right card.
# NO horizontal flip: a mirrored card looks like a different card.
ROBUST_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    _GlareTransform(p=0.40),
    _MotionBlurTransform(p=0.25),
    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4,
                                 saturation=0.4, hue=0.10)], p=0.80),
    T.RandomGrayscale(p=0.05),
    T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.30),
    T.RandomPerspective(distortion_scale=0.30, p=0.50),
    T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.85, 1.15)),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    T.RandomErasing(p=0.30, scale=(0.02, 0.15)),   # simulates finger / partial occlusion
])

CLEAN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


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
    """
    Loads card images with optional heavy augmentation.

    With USE_ROBUST_AUGMENTATION=True the model sees glare, blur, perspective
    distortion, colour jitter, and random erasing — making it invariant to the
    real-world corruptions a phone camera introduces.
    """

    def __init__(self, root, augment=USE_ROBUST_AUGMENTATION):
        self.folder    = dset.ImageFolder(root=root)
        self.imgs      = self.folder.imgs
        self.transform = ROBUST_TRANSFORM if augment else CLEAN_TRANSFORM

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        img    = Image.open(path).convert("RGB")
        img    = imgArtCropper(img)           # crop card art region first
        tensor = self.transform(img)          # (3, 224, 224)
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
