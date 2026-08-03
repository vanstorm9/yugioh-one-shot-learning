# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import pickle
from time import time

from yugioh_utils import (
    DIM, expand_img_dim, imgArtCropper,
    imgPathToTensor, imgPathToNormalizedTensor,
    imgPathToCVImg, imgPathToNormalizedCVImg,
    imgPathToTranslateCVImg, imgPathToNormalizedTranslateCVImg,
    cvImgToTensor, calculateHOGpoints, SiameseNetwork,
)

useGPU = True
enableEval = False

targetDirName = './cardDatabaseFull/'
assert os.path.exists(targetDirName)

loadPath = './savedModels/res-withShift-150-072020.pth'
assert os.path.exists(loadPath)

dictPath = './savedModels/featureMap-withShift-150-072020.pkl'
assert os.path.exists(dictPath)

viewNCards = 3
siftNLim = 500
HOGRankingEnable = False
extremeSIFT = True

dim = DIM
visualizeResult = True

groundTruthPath = targetDirName + 'BlueEyes-White-Dragon-1-89631139/896311391.jpg'
imagePath0 = './test-input/bewd_dd.png'

assert os.path.exists(imagePath0)
assert os.path.exists(groundTruthPath)

cardName = groundTruthPath.split('/')[2]


net = SiameseNetwork().cuda()
net = nn.DataParallel(net, device_ids=[0])
net.load_state_dict(torch.load(loadPath))
if enableEval:
    net.eval()


def calculateSIFTscore(similarScore, numOfPts):
    if extremeSIFT:
        return similarScore - (numOfPts**3) / 100000
    return similarScore - (numOfPts**2) / 100000


orb = cv2.ORB_create()

featureMapDict = pickle.load(open(dictPath, 'rb'))
print(dictPath)
print(loadPath)

tmpPath0 = targetDirName + 'Union-Attack-0-60399954/603999540.jpg'
assert os.path.exists(targetDirName)
assert os.path.exists(tmpPath0)

img0 = imgPathToTensor(tmpPath0)
with torch.no_grad():
    output1, _, _ = net(img0.cuda(), img0.cuda(), img0.cuda())

cardNameTemp = tmpPath0.split('/')[-2]
name, output2 = featureMapDict[cardNameTemp]
euclidean_distance = F.pairwise_distance(output1, output2)
print(euclidean_distance.item())
assert euclidean_distance.item() < 0.001
print('No model / dictionary mismatch')


rankList = []
begin = time()

img0Display = imgPathToTensor(imagePath0)
img0 = imgPathToNormalizedTensor(imagePath0)

siftImg0 = imgArtCropper(cv2.imread(imagePath0, 0))
siftImg0 = cv2.resize(siftImg0, dim, interpolation=cv2.INTER_AREA)
siftImg0 = cv2.equalizeHist(siftImg0)

print(imagePath0)
print(img0.shape)

with torch.no_grad():
    output1, _, _ = net(img0.cuda(), img0.cuda(), img0.cuda())

print('Single image prediction phase: ', time()-begin, 's')
begin = time()

for key, value in featureMapDict.items():
    absPath, output2 = value
    euclidean_distance = F.pairwise_distance(output1, output2)
    rankList.append((key, euclidean_distance.item(), absPath, None, euclidean_distance.item()))

rankList.sort(key=lambda x: x[1])

rankListFinal = []
if HOGRankingEnable:
    for name, score, absPath, _, _ in rankList[:siftNLim]:
        img1 = imgArtCropper(cv2.imread(absPath, 0))
        img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
        numPoints = calculateHOGpoints(orb, siftImg0, img1)
        finalScore = calculateSIFTscore(score, numPoints)
        rankListFinal.append((name, finalScore, absPath, numPoints, score))
    rankListFinal.sort(key=lambda x: x[1])
else:
    rankListFinal = rankList

print('Dictionary manage', time()-begin, 's')

rankStr = ':   rankOriginal:'
if HOGRankingEnable:
    rankStr = ':   rank:'
targetRank = rankStr

guessedCorrectlyRankOne = False
rankIter = 0
incorrectList = []
pred = []

for i, (name, score, absPath, numHOGPoints, originalScore) in enumerate(rankListFinal):
    if absPath.split('/')[2] == cardName:
        if rankIter == 0:
            print('\n\nCorrect\n\n')
            pred = np.append(pred, 1)
            guessedCorrectlyRankOne = True

        img1 = imgPathToTensor(absPath)
        targetRank = ('         ' + name + rankStr + str(i) + ' (out of ' + str(len(rankList)) + ')'
                      '        score:' + str(score) + '          numORBpts:' + str(numHOGPoints)
                      + '        originalScore: ' + str(originalScore))
        if visualizeResult and i < viewNCards:
            print('______________________________________________________________________________________________________')
            print('   ', imagePath0)
            print(targetRank)
            print('______________________________________________________________________________________________________')
    elif i < viewNCards:
        incorrectList.append(name)
        if visualizeResult:
            print(name, ': ', rankStr, i, '     score:', score,
                  '          numORBpts:', str(numHOGPoints), '        originalScore: ', str(originalScore))
    rankIter += 1

if not guessedCorrectlyRankOne:
    print('\n\nIncorrect\n\n')
    pred = np.append(pred, 0)
    for guessedNames in incorrectList:
        print(guessedNames)

if visualizeResult:
    print('\n\n\n')
    print(targetRank)
    print('           ', imagePath0)

correctCount = np.where(pred == 1)[0].shape[0]
wrongCount = np.where(pred == 0)[0].shape[0]
print('Correct: ', correctCount, '        Incorrect count: ', wrongCount)
accuracy = np.where(pred == 1)[0].shape[0] / pred.shape[0]
print('Accuracy: ', accuracy)
