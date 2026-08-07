# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import models_lpf.resnet as mod_res

DIM = (244, 244)


def expand_img_dim(img, numOfExp):
    for _ in range(numOfExp):
        img = np.expand_dims(img, axis=0)
    return img


def imgArtCropper(img):
    if type(img) is np.ndarray:
        width, height = img.shape
        img = img[int(0.2*height):int(0.7*height), int(0.2*width):int(0.8*width)]
    else:
        width, height = img.size
        img = img.crop((int(0.2*width), int(0.2*height), int(0.8*width), int(0.7*height)))
    return img


def imgPathToTensor(absPath, dim=DIM):
    img = imgArtCropper(cv2.resize(cv2.imread(absPath, 0), dim, interpolation=cv2.INTER_AREA)) / 255.0
    img = [img] * 3
    img = expand_img_dim(img, 1)
    return torch.from_numpy(img).type('torch.FloatTensor')


def imgPathToNormalizedTensor(absPath, dim=DIM):
    img = imgArtCropper(cv2.resize(cv2.imread(absPath, 0), dim, interpolation=cv2.INTER_AREA))
    img = cv2.equalizeHist(img)
    img = [img] * 3
    img = expand_img_dim(img, 1)
    return torch.from_numpy(img).type('torch.FloatTensor')


def imgPathToCVImg(absPath, dim=DIM):
    img = imgArtCropper(cv2.resize(cv2.imread(absPath, 0), dim, interpolation=cv2.INTER_AREA)) / 255.0
    return img


def imgPathToNormalizedCVImg(absPath, dim=DIM):
    img = imgArtCropper(cv2.resize(cv2.imread(absPath, 0), dim, interpolation=cv2.INTER_AREA))
    return cv2.equalizeHist(img)


def imgPathToTranslateCVImg(absPath, translation_matrix, dim=DIM):
    img = cv2.resize(cv2.imread(absPath, 0), dim, interpolation=cv2.INTER_AREA)
    img = cv2.warpAffine(img, translation_matrix, (img.shape[0], img.shape[1]))  # fixed: was img1.shape
    img = imgArtCropper(img) / 255.0
    return img


def imgPathToNormalizedTranslateCVImg(absPath, translation_matrix, dim=DIM):
    img = cv2.resize(cv2.imread(absPath, 0), dim, interpolation=cv2.INTER_AREA)
    img = cv2.warpAffine(img, translation_matrix, (img.shape[0], img.shape[1]))
    img = imgArtCropper(img)
    img = cv2.equalizeHist(img)
    return img


def cvImgToTensor(img):
    if img.ndim < 3:
        img = [img] * 3
        img = expand_img_dim(img, 1)
    elif img.ndim != 3:
        print("Image dimensions neither 1 nor 3 (cvImgToTensor)")
        return None
    return torch.from_numpy(img).type('torch.FloatTensor')


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class SiameseNetwork(nn.Module):
    """Inference network using ResNet101-LPF backbone."""
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = mod_res.resnet101(filter_size=3)

    def forward_once(self, x):
        return self.resnet(x)

    def forward(self, input1, input2, input3):
        return self.forward_once(input1), self.forward_once(input2), self.forward_once(input3)


def calculateHOGpoints(orb, img1, img2):
    kpts1, descs1 = orb.detectAndCompute(img1, None)
    kpts2, descs2 = orb.detectAndCompute(img2, None)
    if descs2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs1, descs2)
    return len(matches)


# ---------------------------------------------------------------------------
# DINOv2 utilities
# ---------------------------------------------------------------------------
# transformers must be imported at module level BEFORE faiss is ever imported
# anywhere in the process — they share conflicting BLAS libraries on macOS.
from transformers import AutoModel as _AutoModel  # noqa: F401  (preload)
import torchvision.transforms as T
from PIL import Image

DINO_DIM = (224, 224)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_dino_transform = T.Compose([
    T.Resize(DINO_DIM),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def imgPathToRGBTensor(absPath):
    """Load image, crop card art, return (1,3,224,224) float tensor for DINOv2."""
    img = Image.open(absPath).convert("RGB")
    img = imgArtCropper(img)
    return _dino_transform(img).unsqueeze(0)


def imgArrayToRGBTensor(bgr_np):
    """Convert a BGR numpy crop (from cv2/YOLO) to a (1,3,224,224) DINOv2 tensor."""
    img = Image.fromarray(cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB))
    img = imgArtCropper(img)
    return _dino_transform(img).unsqueeze(0)


class DINOv2Extractor:
    """
    DINOv2-base feature extractor. Returns L2-normalised CLS embeddings.

    Pass lora_checkpoint=<path> to load a fine-tuned LoRA checkpoint produced
    by finetune_arcface.py instead of using the frozen base model.
    """

    def __init__(self, device=None, lora_checkpoint=None):
        from transformers import AutoModel
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" \
                if torch.backends.mps.is_available() else "cpu"
        self.device = device

        backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        if lora_checkpoint is not None:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1, bias="none",
            )
            backbone = get_peft_model(backbone, lora_cfg)
            ckpt = torch.load(lora_checkpoint, map_location=device)
            backbone.load_state_dict(ckpt["backbone"])

        self.model = backbone.eval().to(device)

    @torch.no_grad()
    def extract(self, tensor):
        """tensor: (B,3,224,224) float — returns (B, 768) numpy float32, L2-normalised."""
        tensor = tensor.to(self.device)
        outputs = self.model(pixel_values=tensor)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = F.normalize(cls, p=2, dim=1)
        return cls.cpu().numpy()

    @torch.no_grad()
    def extract_path(self, absPath):
        """Load one image from disk, return (768,) numpy array."""
        tensor = imgPathToRGBTensor(absPath).to(self.device)
        outputs = self.model(pixel_values=tensor)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = F.normalize(cls, p=2, dim=1)
        return cls.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_array(self, bgr_np):
        """Extract from a BGR numpy array (e.g. YOLO crop). Returns (768,) numpy array."""
        tensor = imgArrayToRGBTensor(bgr_np).to(self.device)
        outputs = self.model(pixel_values=tensor)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = F.normalize(cls, p=2, dim=1)
        return cls.squeeze(0).cpu().numpy()
