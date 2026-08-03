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
