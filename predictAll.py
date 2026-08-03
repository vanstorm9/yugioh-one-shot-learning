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
from collections import OrderedDict

from yugioh_utils import (
    DIM, expand_img_dim, imgArtCropper,
    imgPathToTensor, imgPathToNormalizedTensor,
    calculateHOGpoints, SiameseNetwork, TripletLoss,
)

useGPU = True

targetDirName = '../cardDatabaseFull/'
assert os.path.exists(targetDirName)

loadPath = 'savedModels/res-withShift-150-072020.pth'
dictPath = 'savedModels/featureMap-withShift-150-072020.pkl'

assert os.path.exists(dictPath)
assert os.path.exists(loadPath)

viewNCards = 3
siftNLim = 10
HOGRankingEnable = True
extremeSIFT = False

dim = DIM
visualizeResult = True

groundTruthPath = targetDirName + 'Toon-Dark-Magician-0-21296502/21296502.jpg'
imagePath0 = './test-input/tdm_0.png'
assert os.path.exists(imagePath0)

cardName = imagePath0.split('/')[2]


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


folder_dataset = dset.ImageFolder(root='../cardDatabaseFull/')

siamese_dataset = SiameseNetworkDataset(
    imageFolderDataset=folder_dataset,
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((244, 244)),
        transforms.ColorJitter(brightness=(0.2, 1.5), contrast=(0.1, 2.5), hue=.05, saturation=(.0, .15)),
        transforms.RandomAffine(0, translate=(0.0, 0.0), scale=None, shear=(0.0, 0.0),
                                resample=PIL.Image.NEAREST, fillcolor=(0, 0, 0)),
        transforms.ToTensor(),
    ]),
    should_invert=False)

print('Loading train dataloader. . .')
train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=0,
                              batch_size=48)

margin = 2.
criterion = TripletLoss(margin)

if useGPU:
    net = SiameseNetwork().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    net = nn.DataParallel(net, device_ids=[0])
    net.load_state_dict(torch.load(loadPath))
else:
    device = torch.device('cpu')
    net = SiameseNetwork().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    state_dict = torch.load(loadPath, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.` prefix added by DataParallel
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net = net.to(device)

print('Model loaded')


def calculateSIFTscore(similarScore, numOfPts):
    if extremeSIFT:
        return similarScore - (numOfPts**3) / 10000000
    return similarScore - (numOfPts**2) / 100000


def getSimilarRank(imagePath0, imagePath1):
    img0 = imgPathToTensor(imagePath0)
    img1 = imgPathToTensor(imagePath1)
    with torch.no_grad():
        if useGPU:
            output1, output2, _ = net(img0.cuda(), img1.cuda(), img1.cuda())
        else:
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
    if useGPU:
        output1, _, _ = net(img0.cuda(), img0.cuda(), img0.cuda())
    else:
        output1, _, _ = net(img0, img0, img0)

cardNameTemp = tmpPath0.split('/')[-2]
name, output2 = featureMapDict[cardNameTemp]
if useGPU:
    euclidean_distance = F.pairwise_distance(output1, output2)
else:
    euclidean_distance = F.pairwise_distance(output1, output2.detach().cpu())

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

with torch.no_grad():
    if useGPU:
        output1, _, _ = net(img0.cuda(), img0.cuda(), img0.cuda())
    else:
        output1, _, _ = net(img0, img0, img0)

print('Single image prediction phase: ', time()-begin, 's')
begin = time()

for key, value in featureMapDict.items():
    absPath, output2 = value
    if useGPU:
        euclidean_distance = F.pairwise_distance(output1, output2)
    else:
        euclidean_distance = F.pairwise_distance(output1, output2.detach().cpu())
    rankList.append((key, euclidean_distance.item(), absPath, None))

rankList.sort(key=lambda x: x[1])

rankListFinal = []
for name, score, absPath, _ in rankList[:siftNLim]:
    img1 = imgArtCropper(cv2.imread(absPath, 0))
    img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    numPoints = calculateHOGpoints(orb, siftImg0, img1)
    finalScore = calculateSIFTscore(score, numPoints)
    rankListFinal.append((name, finalScore, absPath, numPoints, score))

if HOGRankingEnable:
    rankListFinal.sort(key=lambda x: x[1])
else:
    rankListFinal.sort(key=lambda x: x[4])

print('Dictionary manage', time()-begin, 's')

rankStr = ':   rankOriginal:'
if HOGRankingEnable:
    rankStr = ':   rank:'

guessedCorrectlyRankOne = False
rankIter = 0
incorrectList = []
pred = []
targetRank = ''

for i, (name, score, absPath, numHOGPoints, originalScore) in enumerate(rankListFinal):
    incorrectList.append(name)
    print(cardName, '  ', absPath.split('/')[2])
    if absPath.split('/')[2] == cardName:
        if rankIter == 0:
            print('\n\nCorrect\n\n')
            pred = np.append(pred, 1)
            guessedCorrectlyRankOne = True
        img1 = imgPathToTensor(absPath)
        if visualizeResult:
            targetRank = ('         ' + name + rankStr + str(i) + ' (out of ' + str(len(rankList)) + ')'
                          '        score:' + str(score) + '          numORBpts:' + str(numHOGPoints)
                          + '        originalScore: ' + str(originalScore))
            print('______________________________________________________________________________________________________')
            print('   ', imagePath0)
            print(targetRank)
            print('______________________________________________________________________________________________________')
    elif i < viewNCards:
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
    print('---------------------------------------------------')

accuracy = np.where(pred == 1)[0].shape[0] / pred.shape[0]
print('Accuracy: ', accuracy)
