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
import os
from time import time

import models_lpf.resnet as mod_res
from yugioh_utils import TripletLoss, SiameseNetwork

training_dir = "./cardDatabaseFull/"
train_batch_size = 48
train_number_epochs = 300


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


folder_dataset = dset.ImageFolder(root=training_dir)

siamese_dataset = SiameseNetworkDataset(
    imageFolderDataset=folder_dataset,
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((244, 244)),
        transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.3, 2.0), hue=.05, saturation=(.0, .15)),
        transforms.RandomAffine(0, translate=(0, 0.3), scale=(0.6, 1.8), shear=(0.0, 0.4), resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    should_invert=False)

print('Loading train dataloader. . .')
train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=train_batch_size)

net = SiameseNetwork().cuda()
net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
print('Model parallelized')

margin = 2.
criterion = TripletLoss(margin)
optimizer = optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0
prevNum = -1

for epoch in range(0, train_number_epochs):
    begin = time()
    for i, data in enumerate(train_dataloader, 0):
        img_anc, img_pos, img_neg, _ = data
        img_anc, img_pos, img_neg = img_anc.cuda(), img_pos.cuda(), img_neg.cuda()
        optimizer.zero_grad()
        output1, output2, output3 = net(img_anc, img_pos, img_neg)
        loss = criterion(output1, output2, output3)
        loss.backward()
        optimizer.step()
        if i % 10 == 0 and prevNum != epoch:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss.item())
            prevNum = epoch
    torch.save(net.state_dict(), './res.pth')
    print(time()-begin, 's has passed')

torch.save(net.state_dict(), './res-300-normalized.pth')
