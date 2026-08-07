# -*- coding: utf-8 -*-
"""
Batch evaluation using DINOv2 + FAISS.

Iterates over a random sample of the card database, treating each image as
a query, and measures rank-1 accuracy against the full index.

Usage:
  python evalAll_dino.py
"""

import os
import sys
import json
import argparse
import numpy as np
from time import time

# yugioh_utils pre-loads transformers before faiss to avoid BLAS crash on macOS
from yugioh_utils import DINOv2Extractor, imgPathToRGBTensor, imgArtCropper
import faiss  # must come after yugioh_utils

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description='Evaluate DINOv2+FAISS card recognition')
ap.add_argument('--limit',    type=int, default=None, help='Max cards to evaluate')
ap.add_argument('--index',    default='./savedModels/dino_index.faiss')
ap.add_argument('--meta',     default='./savedModels/dino_meta.json')
ap.add_argument('--card-db',  default='./cardDatabaseFull/')
ap.add_argument('--lora-checkpoint', default=None)
args = ap.parse_args()

INDEX_PATH  = args.index
META_PATH   = args.meta
TARGET_DIR  = args.card_db

assert os.path.exists(INDEX_PATH), f'Run build_faiss_index.py first: {INDEX_PATH}'
assert os.path.exists(META_PATH)
assert os.path.exists(TARGET_DIR)

visualizeResult  = True
testCardIterLim  = args.limit

# ---------------------------------------------------------------------------
# Load index + metadata
# ---------------------------------------------------------------------------
print('Loading FAISS index...')
index = faiss.read_index(INDEX_PATH)
with open(META_PATH) as f:
    meta = json.load(f)
print(f'Index loaded: {index.ntotal} vectors')

extractor = DINOv2Extractor(lora_checkpoint=args.lora_checkpoint)
print('Mode: DINOv2 + FAISS')

# ---------------------------------------------------------------------------
# Collect test images (one per card folder)
# ---------------------------------------------------------------------------
card_folders = sorted(os.listdir(TARGET_DIR))
test_images = []
for folder in card_folders:
    folder_path = os.path.join(TARGET_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if imgs:
        test_images.append((folder, os.path.join(folder_path, sorted(imgs)[0])))

if testCardIterLim:
    test_images = test_images[:testCardIterLim]

print(f'Evaluating on {len(test_images)} cards\n')

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
pred = np.asarray([])
total_begin = time()

for cardTestNum, (cardName, imagePath) in enumerate(test_images):
    loop_begin = time()

    # Embed query
    try:
        query_emb = extractor.extract_path(imagePath)
    except Exception as e:
        print(f'  Skipping {imagePath}: {e}')
        continue

    query_vec = query_emb.reshape(1, -1).astype('float32')
    scores, indices = index.search(query_vec, k=index.ntotal)
    scores  = scores[0]
    indices = indices[0]

    rankList = [(meta[i]['card_name'], float(scores[j]), meta[i]['abs_path'])
                for j, i in enumerate(indices)]

    # Score rank-1
    guessedCorrectlyRankOne = False
    for rankIter, (name, score, absPath) in enumerate(rankList):
        if absPath.split('/')[-2] == cardName:
            if rankIter == 0:
                guessedCorrectlyRankOne = True
                pred = np.append(pred, 1)
                if visualizeResult:
                    print(f'[{cardTestNum}] Correct   {cardName}  '
                          f'rank:0  score:{score:.4f}'
                          f'  ({time()-loop_begin:.2f}s)')
            else:
                pred = np.append(pred, 0)
                if visualizeResult:
                    top1_name = rankList[0][0]
                    print(f'[{cardTestNum}] Incorrect {cardName}  '
                          f'rank:{rankIter}  top1:{top1_name}  score:{score:.4f}'
                          f'  ({time()-loop_begin:.2f}s)')
            break
    else:
        pred = np.append(pred, 0)
        if visualizeResult:
            print(f'[{cardTestNum}] Not found {cardName}')

    correctCount = int(np.where(pred == 1)[0].shape[0])
    total        = int(pred.shape[0])
    print(f'  Running accuracy: {correctCount}/{total} = {correctCount/total:.3f}')
    print('---------------------------------------------------')

# ---------------------------------------------------------------------------
# Final results
# ---------------------------------------------------------------------------
elapsed = time() - total_begin
correctCount = int(np.where(pred == 1)[0].shape[0])
wrongCount   = int(np.where(pred == 0)[0].shape[0])
accuracy     = correctCount / pred.shape[0]

print(f'\nFinal Results')
print(f'  Correct:   {correctCount}')
print(f'  Incorrect: {wrongCount}')
print(f'  Accuracy:  {accuracy:.4f}')
print(f'  Total time: {elapsed:.1f}s  ({elapsed/len(test_images):.2f}s/card)')
