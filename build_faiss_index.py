# -*- coding: utf-8 -*-
"""
Build a FAISS flat-IP index from DINOv2 embeddings of the card database.

Outputs:
  savedModels/dino_index.faiss   — FAISS IndexFlatIP (cosine via L2-norm'd vectors)
  savedModels/dino_meta.json     — [{card_name, abs_path}, ...] parallel to index rows

Usage:
  python build_faiss_index.py
"""

import os
import json
import numpy as np
import torch
from time import time

from yugioh_utils import DINOv2Extractor, imgPathToRGBTensor
import faiss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_DIR   = './cardDatabaseFull/'
OUTPUT_INDEX = './savedModels/dino_index.faiss'
OUTPUT_META  = './savedModels/dino_meta.json'
BATCH_SIZE   = 32
EMBEDDING_DIM = 768   # dinov2-base CLS token

# Set to path of finetune_arcface.py checkpoint to build a LoRA-enhanced index.
# Use the same checkpoint you pass to recognize.py --lora-checkpoint.
# Leave None to use the frozen base DINOv2 model.
LORA_CHECKPOINT = None   # e.g. './savedModels/dino_lora_arcface.pth'

assert os.path.exists(TARGET_DIR), f"Card database not found: {TARGET_DIR}"
os.makedirs('./savedModels', exist_ok=True)

# ---------------------------------------------------------------------------
# Collect all image paths
# ---------------------------------------------------------------------------
image_paths = []
for subdir, _, files in os.walk(TARGET_DIR):
    for f in sorted(files):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(subdir, f))

print(f'Found {len(image_paths)} images in {TARGET_DIR}')

# ---------------------------------------------------------------------------
# Extract embeddings in batches
# ---------------------------------------------------------------------------
extractor = DINOv2Extractor(lora_checkpoint=LORA_CHECKPOINT)
label = f'DINOv2-base + LoRA ({LORA_CHECKPOINT})' if LORA_CHECKPOINT else 'DINOv2-base (frozen)'
print(f'{label} loaded on {extractor.device}')

all_embeddings = []
meta = []

begin = time()
for batch_start in range(0, len(image_paths), BATCH_SIZE):
    batch_paths = image_paths[batch_start:batch_start + BATCH_SIZE]

    tensors = []
    valid_paths = []
    for p in batch_paths:
        try:
            tensors.append(imgPathToRGBTensor(p))
            valid_paths.append(p)
        except Exception as e:
            print(f'  Warning: skipping {p}: {e}')

    if not tensors:
        continue

    batch_tensor = torch.cat(tensors, dim=0)           # (B, 3, 224, 224)
    embeddings = extractor.extract(batch_tensor)        # (B, 768) float32, L2-normed

    for emb, path in zip(embeddings, valid_paths):
        card_name = path.split('/')[-2]
        all_embeddings.append(emb)
        meta.append({'card_name': card_name, 'abs_path': path})

    done = min(batch_start + BATCH_SIZE, len(image_paths))
    elapsed = time() - begin
    rate = done / elapsed
    eta = (len(image_paths) - done) / rate if rate > 0 else 0
    print(f'  {done}/{len(image_paths)}  ({elapsed:.1f}s elapsed, ETA {eta:.0f}s)')

print(f'\nExtracted {len(all_embeddings)} embeddings in {time()-begin:.1f}s')

# ---------------------------------------------------------------------------
# Build FAISS index
# ---------------------------------------------------------------------------
vectors = np.stack(all_embeddings, axis=0).astype('float32')  # (N, 768)

index = faiss.IndexFlatIP(EMBEDDING_DIM)   # Inner product == cosine when L2-normed
index.add(vectors)

faiss.write_index(index, OUTPUT_INDEX)
print(f'FAISS index saved to {OUTPUT_INDEX}  ({index.ntotal} vectors)')

with open(OUTPUT_META, 'w') as f:
    json.dump(meta, f)
print(f'Metadata saved to {OUTPUT_META}')
