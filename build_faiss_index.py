# -*- coding: utf-8 -*-
"""
Build a FAISS flat-IP index from DINOv2 embeddings of the card database.

Outputs:
  savedModels/dino_index.faiss   — FAISS IndexFlatIP (cosine via L2-norm'd vectors)
  savedModels/dino_meta.json     — [{card_name, abs_path}, ...] parallel to index rows

Usage:
  python build_faiss_index.py
  python build_faiss_index.py --aug-copies 5   # embed each card 5x with webcam augmentation
"""

import os
import random
import json
import argparse
import numpy as np
import cv2
import torch
from time import time
from PIL import Image, ImageEnhance

from yugioh_utils import DINOv2Extractor, imgPathToRGBTensor, imgArtCropper, _dino_transform
import faiss

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('--card-db',        default='./cardDatabaseFull/')
ap.add_argument('--output-index',   default='./savedModels/dino_index.faiss')
ap.add_argument('--output-meta',    default='./savedModels/dino_meta.json')
ap.add_argument('--batch-size',     type=int, default=32)
ap.add_argument('--aug-copies',     type=int, default=0,
                help='Extra augmented embeddings per image (0 = clean only). '
                     '5 is recommended for webcam robustness.')
ap.add_argument('--lora-checkpoint', default=None)
args = ap.parse_args()

TARGET_DIR    = args.card_db
OUTPUT_INDEX  = args.output_index
OUTPUT_META   = args.output_meta
BATCH_SIZE    = args.batch_size
EMBEDDING_DIM = 768

assert os.path.exists(TARGET_DIR), f"Card database not found: {TARGET_DIR}"
os.makedirs('./savedModels', exist_ok=True)

# ---------------------------------------------------------------------------
# Webcam augmentation pipeline
# Applied to PIL RGB images before art-crop + DINOv2 normalization.
# Replicates: defocus blur, JPEG compression, sleeve glare, color temp, noise.
# ---------------------------------------------------------------------------

def _webcam_augment(img_pil: Image.Image) -> Image.Image:
    img = np.array(img_pil)  # RGB uint8

    # Defocus blur — most visible issue in real webcam captures
    if random.random() < 0.55:
        sigma = random.uniform(1.0, 5.0)
        k = int(sigma * 4) * 2 + 1  # ensure odd
        img = cv2.GaussianBlur(img, (k, k), sigma)

    # JPEG re-compression artifacts (webcam always compresses)
    if random.random() < 0.65:
        quality = random.randint(40, 85)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # Card sleeve — semi-transparent whitish overlay
    if random.random() < 0.40:
        alpha = random.uniform(0.04, 0.20)
        sleeve = np.ones_like(img, dtype=np.float32) * 240
        img = np.clip(img.astype(np.float32) * (1 - alpha) + sleeve * alpha, 0, 255).astype(np.uint8)

    # Color temperature shift (room lighting)
    if random.random() < 0.50:
        img = img.astype(np.float32)
        if random.random() < 0.5:  # warm
            img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(1.05, 1.20), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(0.80, 0.95), 0, 255)
        else:  # cool
            img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(0.80, 0.95), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(1.05, 1.20), 0, 255)
        img = img.astype(np.uint8)

    # Sensor noise
    if random.random() < 0.50:
        noise = np.random.normal(0, random.uniform(3, 18), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img_pil = Image.fromarray(img)

    # Brightness / contrast jitter
    img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.60, 1.40))
    img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.70, 1.30))

    return img_pil


def _pil_to_tensor(img_pil: Image.Image):
    """Art-crop + DINOv2 normalise → (1, 3, 224, 224) tensor."""
    img_pil = imgArtCropper(img_pil.convert("RGB"))
    return _dino_transform(img_pil).unsqueeze(0)


# ---------------------------------------------------------------------------
# Collect all image paths
# ---------------------------------------------------------------------------
image_paths = []
for subdir, _, files in os.walk(TARGET_DIR):
    for f in sorted(files):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(subdir, f))

total_copies = 1 + args.aug_copies
print(f'Found {len(image_paths)} images  ×  {total_copies} copies  =  '
      f'{len(image_paths) * total_copies} vectors')

# ---------------------------------------------------------------------------
# Extract embeddings
# ---------------------------------------------------------------------------
extractor = DINOv2Extractor(lora_checkpoint=args.lora_checkpoint)
label = f'DINOv2-base + LoRA ({args.lora_checkpoint})' if args.lora_checkpoint else 'DINOv2-base (frozen)'
print(f'{label} on {extractor.device}')

all_embeddings = []
meta = []

begin = time()
for batch_start in range(0, len(image_paths), BATCH_SIZE):
    batch_paths = image_paths[batch_start:batch_start + BATCH_SIZE]

    for copy_idx in range(total_copies):
        tensors     = []
        valid_paths = []
        for p in batch_paths:
            try:
                if copy_idx == 0:
                    # Clean original
                    tensors.append(imgPathToRGBTensor(p))
                else:
                    # Webcam-augmented copy
                    img_pil = Image.open(p).convert("RGB")
                    img_pil = _webcam_augment(img_pil)
                    tensors.append(_pil_to_tensor(img_pil))
                valid_paths.append(p)
            except Exception as e:
                print(f'  Warning: skipping {p}: {e}')

        if not tensors:
            continue

        batch_tensor = torch.cat(tensors, dim=0)
        embeddings   = extractor.extract(batch_tensor)

        for emb, path in zip(embeddings, valid_paths):
            card_name = path.split('/')[-2]
            all_embeddings.append(emb)
            meta.append({'card_name': card_name, 'abs_path': path})

    done    = min(batch_start + BATCH_SIZE, len(image_paths))
    elapsed = time() - begin
    rate    = done / elapsed if elapsed > 0 else 1
    eta     = (len(image_paths) - done) / rate
    print(f'  {done}/{len(image_paths)} images  '
          f'({len(all_embeddings)} vectors so far, ETA {eta:.0f}s)')

print(f'\nExtracted {len(all_embeddings)} embeddings in {time()-begin:.1f}s')

# ---------------------------------------------------------------------------
# Build FAISS index
# ---------------------------------------------------------------------------
vectors = np.stack(all_embeddings, axis=0).astype('float32')

index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(vectors)

faiss.write_index(index, OUTPUT_INDEX)
print(f'FAISS index saved to {OUTPUT_INDEX}  ({index.ntotal} vectors)')

with open(OUTPUT_META, 'w') as f:
    json.dump(meta, f)
print(f'Metadata saved to {OUTPUT_META}')
