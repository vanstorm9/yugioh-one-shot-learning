# -*- coding: utf-8 -*-
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from PIL import Image
import torch
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import cv2
import os
import pickle
from time import time

from yugioh_utils import (
    DIM, expand_img_dim, imgArtCropper,
    imgPathToTensor, imgPathToNormalizedTensor,
    calculateHOGpoints, SiameseNetwork, TripletLoss,
)

useGPU = True
enableEval = False

HOGRankingEnable = True
siftNLim = 600
extremeSIFT = True

targetDirName = './cardDatabaseFull/'
assert os.path.exists(targetDirName)

loadPath = './savedModels/res-withShift-150-072020.pth'
assert os.path.exists(loadPath)

dictPath = './savedModels/featureMap-withShift-150-072020.pkl'
assert os.path.exists(dictPath)

viewNCards = 3
dim = DIM
visualizeResult = True

groundTruthPath = targetDirName + 'Dark-Magician-0-46986414/469864140.jpg'
imagePath0 = './data/cards/training/dark-magician/1.jpg'
assert os.path.exists(imagePath0)

cardName = groundTruthPath.split('/')[2]


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        while True:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] == img1_tuple[1]:
                break

        while True:
            img2_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] != img2_tuple[1]:
                break

        width, height = (244, 244)
        pathList = [(img0_tuple[0], img1_tuple[0], img2_tuple[0])]

        img0 = Image.open(img0_tuple[0]).resize((width, height))
        img1 = Image.open(img1_tuple[0]).resize((width, height))
        img2 = Image.open(img2_tuple[0]).resize((width, height))

        img0 = img0.crop((int(0.2*width), int(0.2*height), int(0.8*width), int(0.7*height)))
        img1 = img1.crop((int(0.2*width), int(0.2*height), int(0.8*width), int(0.7*height)))
        img2 = img2.crop((int(0.2*width), int(0.2*height), int(0.8*width), int(0.7*height)))

        img0 = img0.convert("L")
        img1 = img1.convert("L")
        img2 = img2.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)
            img2 = PIL.ImageOps.invert(img2)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img0, img1, img2, pathList

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


folder_dataset = dset.ImageFolder(root=targetDirName)

siamese_dataset = SiameseNetworkDataset(
    imageFolderDataset=folder_dataset,
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((244, 244)),
        transforms.ColorJitter(brightness=(0.2, 1.5), contrast=(0.1, 2.5), hue=.05, saturation=(.0, .15)),
        transforms.RandomAffine(0, translate=(0.2, 0.2), scale=None, shear=(0.2, 0.2),
                                resample=PIL.Image.NEAREST, fillcolor=(0, 0, 0)),
        transforms.ToTensor(),
    ]),
    should_invert=False)

net = SiameseNetwork().cuda()
margin = 2.
criterion = TripletLoss(margin)
optimizer = optim.Adam(net.parameters(), lr=0.0005)
net = nn.DataParallel(net, device_ids=[0])
net.load_state_dict(torch.load(loadPath))
if enableEval:
    net.eval()


def calculateSIFTscore(similarScore, numOfPts):
    if extremeSIFT:
        return similarScore - (numOfPts**3) / 100000
    return similarScore - (numOfPts**2) / 100000


def getSimilarRank(imagePath0, imagePath1):
    img0 = imgPathToTensor(imagePath0)
    img1 = imgPathToTensor(imagePath1)
    with torch.no_grad():
        output1, output2, _ = net(img0, img1, img1)
    euclidean_distance = F.pairwise_distance(output1, output2)
    return euclidean_distance, (output1, output2)


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


orb = cv2.ORB_create()

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


########################################################################
# Evaluation on test set

class TestDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def imgPathToNormalizedCVImg(self, absPath):
        img0 = imgArtCropper(cv2.resize(cv2.imread(absPath, 0), dim, interpolation=cv2.INTER_AREA))
        img0 = cv2.equalizeHist(img0)
        return img0

    def imgPathToNormalizedTensor(self, absPath):
        img0 = self.imgPathToNormalizedCVImg(absPath)
        img0 = [img0] * 3
        img0 = expand_img_dim(img0, 1)
        return torch.from_numpy(img0).type('torch.FloatTensor')

    def __getitem__(self, index):
        img_tuple = random.choice(self.imageFolderDataset.imgs)
        inputPath = img_tuple[0]
        img0 = Image.fromarray(self.imgPathToNormalizedCVImg(inputPath))
        img0 = self.transform(img0)
        return img0, inputPath

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


testCardIterLim = len(os.listdir(targetDirName))

eval_dataset = TestDataset(
    imageFolderDataset=folder_dataset,
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((244, 244)),
        transforms.ColorJitter(brightness=(0.2, 0.9), contrast=(0.1, 2.0), hue=.05, saturation=(.0, .15)),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=None, shear=None,
                                resample=False, fillcolor=(0, 0, 0)),
        transforms.ToTensor(),
    ]),
    should_invert=False)

test_dataloader = DataLoader(eval_dataset, num_workers=6, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)

pred = np.asarray([])

for cardTestNum in range(0, testCardIterLim):
    rankList = []
    print(cardTestNum)
    x0, dataPathTuple = next(dataiter)
    dataPath = dataPathTuple[0]

    imagePath0 = dataPath
    cardName = imagePath0.split('/')[2]
    print(dataPath)

    begin = time()

    x0 = np.asarray(x0)
    x0 = torch.from_numpy(x0).type('torch.FloatTensor')

    print(x0.shape)

    siftImg0 = imgArtCropper(cv2.imread(dataPath, 0))
    siftImg0 = cv2.resize(siftImg0, dim, interpolation=cv2.INTER_AREA)

    with torch.no_grad():
        output1, _, _ = net(x0.cuda(), x0.cuda(), x0.cuda())

    for key, value in featureMapDict.items():
        absPath, output2 = value
        euclidean_distance = F.pairwise_distance(output1, output2)
        rankList.append((key, euclidean_distance.item(), absPath, None))

    rankList.sort(key=lambda x: x[1])

    if HOGRankingEnable:
        rankListFinal = []
        for name, score, absPath, _ in rankList[:siftNLim]:
            img1 = imgArtCropper(cv2.imread(absPath, 0))
            img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
            numPoints = calculateHOGpoints(orb, siftImg0, img1)
            finalScore = calculateSIFTscore(score, numPoints)
            rankListFinal.append((name, finalScore, absPath, numPoints, score))
        rankListFinal.sort(key=lambda x: x[1])
    else:
        rankListFinal = [(n, s, p, None, s) for n, s, p, _ in rankList]

    if visualizeResult:
        print('Dictionary manage', time()-begin, 's')

    guessedCorrectlyRankOne = False
    rankIter = 0
    incorrectList = []

    for i, (name, score, absPath, numHOGPoints, originalScore) in enumerate(rankListFinal):
        incorrectList.append(name)
        if absPath.split('/')[2] == cardName:
            if rankIter == 0:
                print('\n\nCorrect\n\n')
                pred = np.append(pred, 1)
                guessedCorrectlyRankOne = True
            if visualizeResult:
                rankStr = ('         ' + name + ':   rank:' + str(i) + ' (out of ' + str(len(rankList)) + ')'
                           '        score:' + str(score) + '          numORBpts:' + str(numHOGPoints)
                           + '        originalScore: ' + str(originalScore))
                print('______________________________________________________________________________________________________')
                print('   ', imagePath0)
                print(rankStr)
                print('______________________________________________________________________________________________________')
        elif i < viewNCards:
            if visualizeResult:
                print(name, ': ', i, '     score:', score,
                      '          numORBpts:', str(numHOGPoints), '        originalScore: ', str(originalScore))
        rankIter += 1

    if not guessedCorrectlyRankOne:
        print('\n\nIncorrect\n\n')
        pred = np.append(pred, 0)
        for guessedNames in incorrectList[:10]:
            print(guessedNames)

    correctCount = np.where(pred == 1)[0].shape[0]
    wrongCount = np.where(pred == 0)[0].shape[0]
    print('Correct: ', correctCount, '        Incorrect count: ', wrongCount)
    print('---------------------------------------------------')

accuracy = np.where(pred == 1)[0].shape[0] / pred.shape[0]
print('Accuracy: ', accuracy)
