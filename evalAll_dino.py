# -*- coding: utf-8 -*-
"""
Batch evaluation using DINOv2 + FAISS + optional ORB re-ranking.

Iterates over a random sample of the card database, treating each image as
a query, and measures rank-1 accuracy against the full index.

Usage:
  python evalAll_dino.py
"""

import os
import json
import random
import numpy as np
import torch
import cv2
import faiss
from time import time

from yugioh_utils import (
    DINOv2Extractor, imgPathToRGBTensor,
    imgArtCropper, calculateHOGpoints, DIM,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INDEX_PATH  = './savedModels/dino_index.faiss'
META_PATH   = './savedModels/dino_meta.json'
TARGET_DIR  = './cardDatabaseFull/'

assert os.path.exists(INDEX_PATH), f'Run build_faiss_index.py first: {INDEX_PATH}'
assert os.path.exists(META_PATH)
assert os.path.exists(TARGET_DIR)

HOGRankingEnable = True
siftNLim         = 500
extremeSIFT      = True
dim              = DIM
visualizeResult  = True

# How many cards to evaluate (None = all)
testCardIterLim  = None


def calculateSIFTscore(cosine_score, numOfPts):
    """Higher is better."""
    if extremeSIFT:
        return cosine_score + (numOfPts ** 3) / 100000
    return cosine_score + (numOfPts ** 2) / 100000


# ---------------------------------------------------------------------------
# Load index + metadata
# ---------------------------------------------------------------------------
print('Loading FAISS index...')
index = faiss.read_index(INDEX_PATH)
with open(META_PATH) as f:
    meta = json.load(f)
print(f'Index loaded: {index.ntotal} vectors')

extractor = DINOv2Extractor()
orb = cv2.ORB_create()

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

    # ORB re-ranking
    siftImg0 = imgArtCropper(cv2.imread(imagePath, 0))
    siftImg0 = cv2.resize(siftImg0, dim, interpolation=cv2.INTER_AREA)
    siftImg0 = cv2.equalizeHist(siftImg0)

    if HOGRankingEnable:
        rankListFinal = []
        for name, cosine_score, absPath in rankList[:siftNLim]:
            img1 = imgArtCropper(cv2.imread(absPath, 0))
            img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
            numPoints = calculateHOGpoints(orb, siftImg0, img1)
            finalScore = calculateSIFTscore(cosine_score, numPoints)
            rankListFinal.append((name, finalScore, absPath, numPoints, cosine_score))
        rankListFinal.sort(key=lambda x: x[1], reverse=True)
    else:
        rankListFinal = [(n, s, p, None, s) for n, s, p in rankList]

    # Score rank-1
    guessedCorrectlyRankOne = False
    for rankIter, (name, score, absPath, numHOGPoints, originalScore) in enumerate(rankListFinal):
        if absPath.split('/')[-2] == cardName:
            if rankIter == 0:
                guessedCorrectlyRankOne = True
                pred = np.append(pred, 1)
                if visualizeResult:
                    print(f'[{cardTestNum}] Correct   {cardName}  '
                          f'rank:0  score:{score:.4f}  cosine:{originalScore:.4f}'
                          f'  ({time()-loop_begin:.2f}s)')
            else:
                pred = np.append(pred, 0)
                if visualizeResult:
                    top1_name = rankListFinal[0][0]
                    print(f'[{cardTestNum}] Incorrect {cardName}  '
                          f'rank:{rankIter}  top1:{top1_name}  cosine:{originalScore:.4f}'
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
