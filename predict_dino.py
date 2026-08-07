# -*- coding: utf-8 -*-
"""
Single-image card prediction using DINOv2 + FAISS.

Usage:
  python predict_dino.py
"""

import os
import json
import numpy as np
from time import time

from yugioh_utils import DINOv2Extractor, imgPathToRGBTensor, imgArtCropper
import faiss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INDEX_PATH  = './savedModels/dino_index.faiss'
META_PATH   = './savedModels/dino_meta.json'

# Ground truth for evaluation
TARGET_DIR       = './cardDatabaseFull/'
groundTruthPath  = TARGET_DIR + 'BlueEyes-White-Dragon-1-89631139/896311391.jpg'
imagePath0       = './test-input/bewd_dd.png'

assert os.path.exists(INDEX_PATH), f'Run build_faiss_index.py first: {INDEX_PATH}'
assert os.path.exists(META_PATH)
assert os.path.exists(imagePath0)
assert os.path.exists(groundTruthPath)

cardName    = groundTruthPath.split('/')[2]
viewNCards  = 3
visualizeResult = True

# ---------------------------------------------------------------------------
# Load index + metadata
# ---------------------------------------------------------------------------
print('Loading FAISS index...')
index = faiss.read_index(INDEX_PATH)
with open(META_PATH) as f:
    meta = json.load(f)
print(f'Index loaded: {index.ntotal} vectors')

extractor = DINOv2Extractor()

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
begin = time()

query_emb = extractor.extract_path(imagePath0)          # (768,) L2-normed
query_vec = query_emb.reshape(1, -1).astype('float32')

scores, indices = index.search(query_vec, k=index.ntotal)  # exact search over all
scores   = scores[0]    # (N,) higher = more similar
indices  = indices[0]   # (N,)

print(f'Embedding + search: {time()-begin:.3f}s')

# Build ranked list: (card_name, cosine_score, abs_path)
rankList = [(meta[i]['card_name'], float(scores[j]), meta[i]['abs_path'])
            for j, i in enumerate(indices)]
# Already sorted descending by cosine score from FAISS

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
guessedCorrectlyRankOne = False
incorrectList = []
pred = []
targetRank = ''

for i, (name, score, absPath) in enumerate(rankList):
    if absPath.split('/')[2] == cardName:
        if i == 0:
            print('\n\nCorrect\n\n')
            pred = np.append(pred, 1)
            guessedCorrectlyRankOne = True
        targetRank = (f'         {name}  rank:{i} (out of {len(rankList)})'
                      f'  score:{score:.4f}')
        if visualizeResult and i < viewNCards:
            print('_' * 90)
            print(f'   {imagePath0}')
            print(targetRank)
            print('_' * 90)
    elif i < viewNCards:
        incorrectList.append(name)
        if visualizeResult:
            print(f'{name}  rank:{i}  score:{score:.4f}')

if not guessedCorrectlyRankOne:
    print('\n\nIncorrect\n\n')
    pred = np.append(pred, 0)
    for g in incorrectList:
        print(g)

if visualizeResult:
    print(f'\n{targetRank}')
    print(f'           {imagePath0}')

correctCount = int(np.where(pred == 1)[0].shape[0])
wrongCount   = int(np.where(pred == 0)[0].shape[0])
print(f'Correct: {correctCount}        Incorrect: {wrongCount}')
print(f'Accuracy: {correctCount / pred.shape[0]:.3f}')
