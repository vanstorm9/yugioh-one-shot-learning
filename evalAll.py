# -*- coding: utf-8 -*-

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import cv2
from matplotlib import pyplot as plt
import os
#from torchsummary import summary

from time import time
import random
import torchvision.models as models
import pickle

import models_lpf.resnet as mod_res


useGPU = True
enableEval = False
#enableEval = True


HOGRankingEnable = True
#HOGRankingEnable = False

siftNLim = 600
#siftNLim = 1000


targetDirName = './cardDatabaseFull/'
assert os.path.exists(targetDirName)

#loadPath = './res-5.pth'
#loadPath = 'resShiftEasy-resnet101-e300-b24.pth'
loadPath = './savedModels/res-withShift-150-072020.pth'
assert os.path.exists(loadPath)


#dictPath = 'featureMap-resShiftEasy-res101-e300-b24.pkl'
#dictPath = 'featureMap-combined.pkl'

#dictPath = 'featureMap-resShiftEasy-res101-e300-b24-eval.pkl'

dictPath = './savedModels/featureMap-withShift-150-072020.pkl'
#dictPath = 'featureMap-5.pkl'
#dictPath = 'featureMap-resL-resnet101-e245-b24-lte.pkl'
assert os.path.exists(dictPath)


viewNCards = 3
#extremeSIFT = False
extremeSIFT = True



dim=(255,255)




visualizeResult = True


# Dark Magician (still problems with sifted-dm_0.png)
#imagePath0 = './test-input/dm_0.png'
groundTruthPath = targetDirName + 'Dark-Magician-0-46986414/469864140.jpg'
imagePath0 = './data/cards/training/dark-magician/1.jpg'
#imagePath0 = './test-input/sifted-dm_0.png'
assert os.path.exists(imagePath0)


cardName = groundTruthPath.split('/')[2]


"""## Helper functions
Set of helper functions
"""

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

"""## Configuration Class
A simple class to manage configuration
"""

class Config():
    #training_dir = "./data/cards_old/training/"
    #testing_dir = "./data/cards_old/testing/"
    
    #training_dir = "./data/cards/training/"
    #testing_dir = "./data/cards/testing/"
    
    training_dir = "./cardDatabaseFull/"
    testing_dir = "./cardDatabaseFull/"
    #testing_dir = "./data/cards_old/testing/"
    
    train_batch_size = 24
    #train_batch_size = 8
    train_number_epochs = 120

"""## Custom Dataset Class
This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair
"""

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self,index):

        # Get an image
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # Get an image from the same class
        while True:
            #keep looping till the same class image is found
            img1_tuple = random.choice(self.imageFolderDataset.imgs) 
            if img0_tuple[1]==img1_tuple[1]:
                break

        # Get an image from a different class
        while True:
            #keep looping till a different class image is found
                
            img2_tuple = random.choice(self.imageFolderDataset.imgs) 
            if img0_tuple[1] !=img2_tuple[1]:
                break

        #width,height = (100,150)
        width,height = (244,244)

        pathList = []
        pathList.append((img0_tuple[0],img1_tuple[0],img2_tuple[0]))

        img0 = Image.open(img0_tuple[0]).resize((width,height))
        img1 = Image.open(img1_tuple[0]).resize((width,height))
        img2 = Image.open(img2_tuple[0]).resize((width,height))
        
        
        # Crop the card art
        #img0 = img0[int(0.2*height):int(0.7*height),int(0.2*width):int(0.8*width)]
        #img1 = img1[int(0.2*height):int(0.7*height),int(0.2*width):int(0.8*width)]
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
        
        #return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))

        # anchor, positive image, negative image
        return img0, img1 , img2, pathList

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

"""## Using Image Folder Dataset"""

folder_dataset = dset.ImageFolder(root=Config.training_dir)

# Commented out IPython magic to ensure Python compatibility.
class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            #iaa.Scale((224, 224)),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            #iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                      iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                 iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)
'''
# https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=aUpukiy8sBKx
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([
                                                    transforms.Grayscale(num_output_channels=3),
                                                    #transforms.Resize((100,100)),
                                                    transforms.Resize((244,244)),
                                                    transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.3,2.0),hue=.05, saturation=(.0,.15)),
                                                    #transforms.RandomHorizontalFlip(),
                                                    #transforms.RandomRotation(10,fill=(0,)),
                                                    
                                                    transforms.RandomAffine(0, translate=(0,0.3), scale=(0.6,1.8), shear=(0.0,0.4), resample=False, fillcolor=0),
                                                    transforms.ToTensor()
                                                ])
                                       ,should_invert=False)
'''

siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([
                                                    transforms.Grayscale(num_output_channels=3),
                                                    transforms.Resize((244,244)),
                                                    transforms.ColorJitter(brightness=(0.2,1.5),contrast=(0.1,2.5),hue=.05, saturation=(.0,.15)),

                                                    transforms.RandomAffine(0, translate=(0.2,0.2), scale=None, shear=(0.2,0.2), resample=PIL.Image.NEAREST, fillcolor=(0,0,0)),
                                                    transforms.ToTensor()
                                                ]),should_invert=False)



vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        #$num_workers=8,
                        num_workers=0,
                        batch_size=8)
dataiter = iter(vis_dataloader)


        
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()


        #self.resnet = models.resnet152(pretrained=True)
        #self.resnet = models.resnet101(pretrained=True)

        self.resnet = mod_res.resnet101(filter_size=3)

        
        

        #self.resnet = models.resnet50(pretrained=True)

        #self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

    def forward_once(self, x):
        '''
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        #print(output.shape)
        #print(output)
        '''
        #begin = time()
        output = self.resnet(x)
        #print('Time for forward prop: ', time()-begin)

        return output
        

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)

        return output1, output2, output3

"""## Contrastive Loss / Triplet Loss"""



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        begin = time()
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        print('Time for contrastive Loss: ', time()-begin)
        return loss_contrastive


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        begin = time()
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        print('Time for triplet loss: ', time()-begin)

        return losses.mean() if size_average else losses.sum()

"""## Training Time!"""

train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        #num_workers=8,
                        num_workers=0,
                        batch_size=Config.train_batch_size)

#net = SiameseNetwork_old().cuda()
#net = SiameseNetwork_old()
net = SiameseNetwork().cuda()
#net = SiameseNetwork()
#net = SiameseNetwork(Bottleneck, [3,4,23,3])
#criterion = ContrastiveLoss()
margin = 2.
criterion = TripletLoss(margin)

optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

#net = nn.DataParallel(net,device_ids=[0,1,2,3])
net = nn.DataParallel(net,device_ids=[0])
#net = nn.DataParallel(net,device_ids)




#net.load_state_dict(torch.load(loadPath,map_location=torch.device('cpu')))
net.load_state_dict(torch.load(loadPath))


if enableEval:
    net.eval()

def expand_img_dim(img,numOfExp):
    for i in range(0,numOfExp):
        img = np.expand_dims(img,axis=0)
    return img

def imgArtCropper(img):
    
    if type(img) is np.ndarray:
        width,height = img.shape
        img = img[int(0.2*height):int(0.7*height),int(0.2*width):int(0.8*width)]
    else:
        width, height = img.size
        img = img.crop((int(0.2*width), int(0.2*height), int(0.8*width), int(0.7*height))) 
        
    return img



def getSimilarRank(imagePath0,imagePath1):
    # Load image 0
    img0 = imgArtCropper(cv2.resize(cv2.imread(imagePath0,0), dim, interpolation = cv2.INTER_AREA))/255.0
    img0 = cv2.resize(img0, dim, interpolation = cv2.INTER_AREA)
    img0 = [img0] * 3
    img0 = expand_img_dim(img0,1)
    img0 = torch.from_numpy(img0).type('torch.FloatTensor')

    # Load image 1
    img1 = imgArtCropper(cv2.resize(cv2.imread(imagePath1,0), dim, interpolation = cv2.INTER_AREA))/255.0
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    img1 = [img1] * 3
    img1 = expand_img_dim(img1,1)
    img1 = torch.from_numpy(img1).type('torch.FloatTensor')

    # Compare and get similary rank
    concatenated = torch.cat((img0,img1),0)
    output1,output2,_ = net(Variable(img0),Variable(img1),Variable(img1))
    #output1,output2,output3 = net(Variable(img0).cuda(),Variable(img1).cuda(),Variable(img1).cuda())
    begin = time()
    euclidean_distance = F.pairwise_distance(output1, output2)
    print('Time to compare euclidean distance: ', time()-begin,'s')
    return euclidean_distance, (output1,output2)

def compareNCards(imgPath0,imgPath1,targetDirName,n_compare=10):
    # Declarations
    rankList = []
    featureMapList = []

    # N-way one shot learning evaluation
    # Compare positive images
    euclidean_distance,featureMap = getSimilarRank(imagePath0,imagePath1)
    output1 = featureMap[0]

    rankList.append((imagePath1.split('/')[-2],euclidean_distance.item(),imagePath1))
    featureMapList.append(output1)


    # Retrieve N negative images
    cardList = os.listdir(targetDirName)
    random.shuffle(cardList)
    cardList = cardList[:n_compare]
    negList = []
    for folderCard in cardList:
      cardDir = targetDirName + folderCard + '/'
      imgTar = os.listdir(cardDir)[0]
      imgPath = cardDir + imgTar
      negList.append(imgPath)


    # Different
    for filePath in negList:
        trueFilePath = filePath 
        euclidean_distance,featureMap = getSimilarRank(imagePath0,trueFilePath)
        output1 = featureMap[0]

        rankList.append((trueFilePath.split('/')[-2],euclidean_distance.item(),trueFilePath))
        #featureMapList.append(output1)

    rankList.sort(key = lambda x: x[1])
    return rankList, featureMapList


# Construct a dictionary of features maps, and save is

targetDirName = './cardDatabaseFull/'

featureMapList = {}
dim = (244,244)

count = 0
limit = 50
limitCnt = 0



#####
# We check for model / dictionary mismatch
#####
featureMapDict = pickle.load(open(dictPath, 'rb'))
print(dictPath)
print(loadPath)


# We do some unit tests to see if model matches dictionary activation maps
tmpPath0 = targetDirName+'Union-Attack-0-60399954/603999540.jpg'

assert os.path.exists(targetDirName)
assert os.path.exists(tmpPath0)



img0 = imgArtCropper(cv2.resize(cv2.imread(tmpPath0,0), dim, interpolation = cv2.INTER_AREA))/255.0
img0 = cv2.resize(img0, dim, interpolation = cv2.INTER_AREA)
img0 = [img0] * 3
img0 = expand_img_dim(img0,1)
img0 = torch.from_numpy(img0).type('torch.FloatTensor')

if useGPU:
    output1,output2,output3 = net(Variable(img0).cuda(),Variable(img0).cuda(),Variable(img0).cuda())
else:
    output1,output2,output3 = net(Variable(img0),Variable(img0),Variable(img0))

cardNameTemp = tmpPath0.split('/')[-2]
name,output2 = featureMapDict[cardNameTemp]
print(cardNameTemp)

if useGPU:
    euclidean_distance = F.pairwise_distance(output1, output2)
else:
    euclidean_distance = F.pairwise_distance(output1, output2.detach().cpu())

print(euclidean_distance.item())

assert euclidean_distance.item() < 0.001
print('No model / dictionary mismatch')










##########################################

def imgPathToCVImg(absPath):
  img = imgArtCropper(cv2.resize(cv2.imread(absPath,0), dim, interpolation = cv2.INTER_AREA))/255.0
  img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  return img

def imgPathToTranslateCVImg(absPath,translation_matrix):
  img = cv2.resize(cv2.imread(absPath,0), dim, interpolation = cv2.INTER_AREA)
  img = cv2.warpAffine(img, translation_matrix, (img1.shape[0],img1.shape[1]))
  img = imgArtCropper(img)/255.0
  img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  return img

def imgPathToNormalizedTranslateCVImg(absPath,translation_matrix):
  img = cv2.resize(cv2.imread(absPath,0), dim, interpolation = cv2.INTER_AREA)
  img = cv2.warpAffine(img, translation_matrix, (img.shape[0],img.shape[1]))
  img = imgArtCropper(img)
  img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  img = cv2.equalizeHist(img)
  return img


def imgPathToNormalizedCVImg(absPath):
  img0 = imgArtCropper(cv2.resize(cv2.imread(absPath,0), dim, interpolation = cv2.INTER_AREA))
  img0 = cv2.resize(img0, dim, interpolation = cv2.INTER_AREA)

  img0 = cv2.equalizeHist(img0)
  return img0

def cvImgToTensor(img):
  if img.ndim < 3:
    img = [img] * 3
    img = expand_img_dim(img,1)
  elif img.ndim != 3:
    print("The dimensions of the image is neither 1 nor 3, this should not happen (cvImgToTensor)")
    return None
  img = torch.from_numpy(img).type('torch.FloatTensor')
  return img

def imgPathToTensor(absPath):
  img1 = imgArtCropper(cv2.resize(cv2.imread(absPath,0), dim, interpolation = cv2.INTER_AREA))/255.0
  img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
  img1 = [img1] * 3
  img1 = expand_img_dim(img1,1)
  img1 = torch.from_numpy(img1).type('torch.FloatTensor')

  return img1


def imgPathToNormalizedTensor(absPath):
    img0 = imgArtCropper(cv2.resize(cv2.imread(absPath,0), dim, interpolation = cv2.INTER_AREA))
    img0 = cv2.resize(img0, dim, interpolation = cv2.INTER_AREA)

    img0 = cv2.equalizeHist(img0)

    img0 = [img0] * 3
    img0 = expand_img_dim(img0,1)
    img0 = torch.from_numpy(img0).type('torch.FloatTensor')
    return img0



def calculateSIFTscore(similarScore, numOfPts):
    if extremeSIFT:
        return similarScore-(numOfPts**3)/10000000
    return similarScore-(numOfPts**2)/100000

def calculateHOGpoints(orb,img1,img2):
  kpts1, descs1 = orb.detectAndCompute(img1,None)
  kpts2, descs2 = orb.detectAndCompute(img2,None)

  if descs2 is None:
      return 0

  #print(len(descs2))

  ## match descriptors and sort them in the order of their distance
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(descs1, descs2)
  dmatches = sorted(matches, key = lambda x:x.distance)

  numOfMatches = len(matches)

  return numOfMatches

orb = cv2.ORB_create()




# Faster way to predict by using dictionary
# We calculate the output of all images in the dataset ahead of time and store them in a dictionary
# We iterate through that and calulate distance score
# This means checking each input's simularity rank takes under 2 seconds to run

# Will consider on using a min heap to make this faster



rankList = []

begin = time()


img0Display = imgPathToTensor(imagePath0) # For display purposes
img0 = imgPathToNormalizedTensor(imagePath0)
#img0 = imgPathToTensor(imagePath0)


#siftImg0 = imgArtCropper(cv2.resize(cv2.imread(imagePath0,0), dim, interpolation = cv2.INTER_AREA))/255.0
siftImg0 = imgArtCropper(cv2.imread(imagePath0,0))
siftImg0 = cv2.resize(siftImg0, dim, interpolation = cv2.INTER_AREA)
siftImg0 = cv2.equalizeHist(siftImg0)




print(imagePath0)
print(img0.shape)

# Get the activation map of the test image
with torch.no_grad():
    if useGPU:
        output1,output2,output3 = net(Variable(img0).cuda(),Variable(img0).cuda(),Variable(img0).cuda())
    else:
        output1,output2,output3 = net(Variable(img0),Variable(img0),Variable(img0))
#output1 = net.forward_once(img0)

print('Single image prediction phase: ',time()-begin,'s')
begin = time()



# Iterating through all activation maps
for key,value in featureMapDict.items():
  
  absPath,output2 = value
  if useGPU:
    euclidean_distance = F.pairwise_distance(output1, output2)
  else:
    euclidean_distance = F.pairwise_distance(output1, output2.detach().cpu())

  #rankList.append((trueFilePath.split('/')[-2],euclidean_distance.item(),trueFilePath))
  rankList.append((key,euclidean_distance.item(),absPath,None, euclidean_distance.item()))




rankList.sort(key = lambda x: x[1])


# Now we just get results
print('\n\n\n\n\n')
i = 0
#for name,score,absPath in rankList:

# Now we calculate for HOG points


rankListFinal = []
for name,score,absPath,_,_ in rankList[:siftNLim]:
  #img1 = imgArtCropper(cv2.resize(cv2.imread(absPath,0), dim, interpolation = cv2.INTER_AREA))/255.0
  img1 = imgArtCropper(cv2.imread(absPath,0))
  img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

  numPoints = calculateHOGpoints(orb,siftImg0,img1)
  finalScore = calculateSIFTscore(score, numPoints)
  rankListFinal.append((name,finalScore,absPath,numPoints,score))

  
if HOGRankingEnable:
  rankListFinal.sort(key = lambda x: x[1])
else:
  rankListFinal = rankList

  
print('Dictionary manage',time()-begin,'s')

rankStr = None

guessedCorrectlyRankOne = False

rankIter = 0
incorrectList = []
pred = []





targetRank = ''

for name,score,absPath,numHOGPoints,originalScore in rankListFinal:



  #print(name,': ', score)
  rankStr = ':   rankOriginal:'
  if HOGRankingEnable:
      rankStr = ':   rank:'


  #print('Comparing: ', cardName, '  ', absPath.split('/')[2])
  # We matched with the correct monster
  if absPath.split('/')[2] == cardName:
    if(rankIter == 0):
      print('\n\nCorrect\n\n')
      pred = np.append(pred,1)
      guessedCorrectlyRankOne = True

    img1 = imgPathToTensor(absPath)

    rankStr = '         ' + name + rankStr + str(i) + ' (out of '+str(len(rankList))+')        score:'+str(score) + '          numORBpts:' + str(numHOGPoints)+'        originalScore: ' + str(originalScore)
    targetRank = rankStr 
    if visualizeResult and i < viewNCards:
      concatenated = torch.cat((img0Display,img1),0)

      print('______________________________________________________________________________________________________')
      print('\n\n\n')

      print('   ',imagePath0)
      print(rankStr)
      print('\n\n')
      print('______________________________________________________________________________________________________')
  elif i < viewNCards:
    incorrectList.append(name)
    img1 = imgPathToTensor(absPath)
    concatenated = torch.cat((img0Display,img1),0)

    if visualizeResult:
    #if visualizeResult or not guessedCorrectlyRankOne:

      print(name,': ',rankStr,i,'     score:',score, '          numORBpts:', str(numHOGPoints),'        originalScore: ', str(originalScore))
  i+=1
  rankIter+=1

if not guessedCorrectlyRankOne:
    print('\n\nIncorrect\n\n')
    pred = np.append(pred,0)

    # Show all of the predictions
    for guessedNames in incorrectList:
      print(guessedNames)

if visualizeResult:
    print('\n\n\n')
    print(targetRank)
    print('           ',imagePath0)
    print('\n\n\n')
    correctCount = np.where(pred==1)[0].shape[0]
    wrongCount = np.where(pred==0)[0].shape[0]
    print('Correct: ', correctCount, '        Incorrect count: ',wrongCount)

    print('---------------------------------------------------')

# Then we show the results
#print(predictList)



correctCount = np.where(pred==1)[0].shape[0]
wrongCount = np.where(pred==0)[0].shape[0]
print('Correct: ', correctCount, '        Incorrect count: ',wrongCount)

accuracy = np.where(pred==1)[0].shape[0]/pred.shape[0]
print('Accuracy: ', accuracy)



########################################################################
# Now we will begin evaluation

rankStr = None

guessedCorrectlyRankOne = False

rankIter = 0
incorrectList = []
pred = []


# Initalize our dataloader
class TestDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert

    
    def imgPathToNormalizedCVImg(self,absPath):
        img0 = imgArtCropper(cv2.resize(cv2.imread(absPath,0), dim, interpolation = cv2.INTER_AREA))
        img0 = cv2.resize(img0, dim, interpolation = cv2.INTER_AREA)

        img0 = cv2.equalizeHist(img0)
        return img0

    def imgPathToNormalizedTensor(self,absPath):
        img0 = imgArtCropper(cv2.resize(cv2.imread(absPath,0), dim, interpolation = cv2.INTER_AREA))
        img0 = cv2.resize(img0, dim, interpolation = cv2.INTER_AREA)

        img0 = cv2.equalizeHist(img0)

        img0 = [img0] * 3
        img0 = expand_img_dim(img0,1)
        img0 = torch.from_numpy(img0).type('torch.FloatTensor')
        return img0

    def __getitem__(self,index):
        width,height = (244,244)

        img_tuple = random.choice(self.imageFolderDataset.imgs)
        inputPath = img_tuple[0]

        
        img0 = self.imgPathToNormalizedCVImg(inputPath)
        
        img0 = Image.fromarray(img0)
        img0 = self.transform(img0)

        return img0, inputPath
        

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
  



# THIS CELL IS FOR EVALUATING ON A TESTSET ###

#testCardIterLim = 5
testCardIterLim = len(os.listdir(targetDirName))



#print('Loading Dataloader. . .')
'''
eval_dataset = TestDataset(imageFolderDataset=folder_dataset,
                    transform=transforms.Compose([
                        transforms.Resize((244,244)),
                        transforms.Grayscale(num_output_channels=3),

                        # Original
                        transforms.ColorJitter(brightness=(0.4,0.9),contrast=(0.1,2.0),hue=.05, saturation=(.0,.15)),
                        transforms.RandomAffine(0, translate=(0,0.2), scale=None, shear=(0.0,0.8), resample=False, fillcolor=(0,0,0)),

                        #transforms.ColorJitter(brightness=(0.4,0.9),contrast=(1.0,2.0),hue=.05, saturation=(.0,.15)),
                        #transforms.RandomAffine(0, translate=(0.1,0.1), scale=None, shear=(0.0,0.8), resample=False, fillcolor=(0,0,0)),

                        #transforms.ColorJitter(brightness=(0.4,0.9),contrast=(1.0,2.0),hue=.05, saturation=(.0,.15)),
                        #transforms.RandomAffine(0, translate=(0.2,0.2), scale=None, shear=(0.0,0.0), resample=False, fillcolor=(0,0,0)),

                        #transforms.ColorJitter(brightness=(0.2,1.0),contrast=(0.1,2.0),hue=.05, saturation=(.0,.15)),
                        #transforms.RandomAffine(0, translate=(0.2,0.2), scale=None, shear=(0.2,0.2), resample=PIL.Image.NEAREST, fillcolor=(0,0,0)),

                        #transforms.ColorJitter(brightness=(0.2,1.8),contrast=(0.1,2.2),hue=.05, saturation=(.0,.15)),
                        #transforms.RandomAffine(0, translate=(0.3,0.3), scale=(0.6,1.8), shear=(0.8,0.8), resample=False, fillcolor=(0,0,0)),

                        transforms.ToTensor(),
                    ]),should_invert=False)
'''
eval_dataset = TestDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([
                                                    transforms.Grayscale(num_output_channels=3),
                                                    transforms.Resize((244,244)),
                                                    #transforms.ColorJitter(brightness=(0.2,1.5),contrast=(0.1,2.5),hue=.05, saturation=(.0,.15)),    # Originally trained dataset

                                                    #transforms.RandomAffine(0, translate=(0.2,0.2), scale=None, shear=(0.2,0.2), resample=PIL.Image.NEAREST, fillcolor=(0,0,0)),
                                                    transforms.ColorJitter(brightness=(0.2,0.9),contrast=(0.1,2.0),hue=.05, saturation=(.0,.15)),

                                                    transforms.RandomAffine(0, translate=(0.1,0.1), scale=None, shear=None, resample=False, fillcolor=(0,0,0)),
                                                    transforms.ToTensor()
                                                ]),should_invert=False)






test_dataloader = DataLoader(eval_dataset,num_workers=6,batch_size=1,shuffle=True)

dataiter = iter(test_dataloader)


predictList = []
pred = np.asarray(predictList)


for cardTestNum in range(0,testCardIterLim):
  rankList = []
  print(cardTestNum)
  x0,dataPathTuple = next(dataiter)
  dataPath = dataPathTuple[0]

  imagePath0 = dataPath
  cardName = imagePath0.split('/')[2]
  #print(dataPath[0][0][0].split('/')[2])
  print(dataPath)

  #print('Loaded Dataloader!')
  begin = time()


  x0 = np.asarray(x0)
  #x0 = [x0] * 3
  #x0 = expand_img_dim(x0,1)
  x0 = torch.from_numpy(x0).type('torch.FloatTensor')

  print(x0.shape)
  #imshow(torchvision.utils.make_grid(x0))
  print('Ground truth: ', dataPath)

  siftImg0 = imgArtCropper(cv2.imread(dataPath,0))
  siftImg0 = cv2.resize(siftImg0, dim, interpolation = cv2.INTER_AREA)

  with torch.no_grad():
   output1,_,_ = net(Variable(x0).cuda(),Variable(x0).cuda(),Variable(x0).cuda())



  # Iterating through all activation maps
  for key,value in featureMapDict.items():
    absPath,output2 = value
    euclidean_distance = F.pairwise_distance(output1, output2)
    #rankList.append((trueFilePath.split('/')[-2],euclidean_distance.item(),trueFilePath))
    rankList.append((key,euclidean_distance.item(),absPath,None))



  rankList.sort(key = lambda x: x[1])


  # Now we just get results

  if visualizeResult:
    print('\n\n\n\n\n')
  i = 0
  #for name,score,absPath in rankList:

  # Now we calculate for HOG points

  if HOGRankingEnable:
    rankListFinal = []
    for name,score,absPath,_ in rankList[:siftNLim]:
      #img1 = imgArtCropper(cv2.resize(cv2.imread(absPath,0), dim, interpolation = cv2.INTER_AREA))/255.0
      img1 = imgArtCropper(cv2.imread(absPath,0))
      img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

      numPoints = calculateHOGpoints(orb,siftImg0,img1)
      finalScore = calculateSIFTscore(score, numPoints)
      rankListFinal.append((name,finalScore,absPath,numPoints,score))

    if HOGRankingEnable:
      rankListFinal.sort(key = lambda x: x[1])
    else:
      rankListFinal = rankList
  else:
    rankListFinal = rankList
  
  if visualizeResult:
    print('Dictionary manage',time()-begin,'s')

  rankStr = None
  guessedCorrectlyRankOne = False

  rankIter = 0
  incorrectList = []
  for name,score,absPath,numHOGPoints,originalScore in rankListFinal:
      
      rankStr = ':   rankOriginal:'
      if HOGRankingEnable:
        rankStr = ':   rank:'


      #print(name,': ', score)
      incorrectList.append(name)
      
      
      #print(cardName, '  ', absPath.split('/')[2])
      # We matched with the correct monster
      if absPath.split('/')[2] == cardName:
        if(rankIter == 0):
          print('\n\nCorrect\n\n')
          pred = np.append(pred,1)
          guessedCorrectlyRankOne = True

        img1 = imgPathToTensor(absPath)

        if visualizeResult:
          concatenated = torch.cat((img0Display,img1),0)
    
          print('______________________________________________________________________________________________________')
          print('\n\n\n')
          
          rankStr = '         ' + name + rankStr + str(i) + ' (out of '+str(len(rankList))+')        score:'+str(score) + '          numORBpts:' + str(numHOGPoints)+'        originalScore: ' + str(originalScore)
          print('   ',imagePath0)
          print(rankStr)
          print('\n\n')
          print('______________________________________________________________________________________________________')
      elif i < viewNCards:
        img1 = imgPathToTensor(absPath)
        concatenated = torch.cat((img0Display,img1),0)

        if visualizeResult:
        #if visualizeResult or not guessedCorrectlyRankOne:
          
          print(name,': ',rankStr,i,'     score:',score, '          numORBpts:', str(numHOGPoints),'        originalScore: ', str(originalScore))
      i+=1
      rankIter+=1

  if not guessedCorrectlyRankOne:
    print('\n\nIncorrect\n\n')
    pred = np.append(pred,0)
    
    # Show all of the predictions
    for guessedNames in incorrectList[:10]:
      print(guessedNames)
    i = 0
    for name,score,absPath,numHOGPoints in rankList:
      if absPath.split('/')[2] == cardName:
          print('Original rank: ')
          rankStr = '         ' + name + rankStr + str(i) + ' (out of '+str(len(rankList))+')            originalScore: ' + str(score)
          print(rankStr)
          print('\n\n')
      i += 1

  if visualizeResult:
    print('\n\n\n')
    print(rankStr)
    print('           ',imagePath0)
    print('\n\n\n')


  
  correctCount = np.where(pred==1)[0].shape[0] 
  wrongCount = np.where(pred==0)[0].shape[0] 
  print('Correct: ', correctCount, '        Incorrect count: ',wrongCount)

  print('---------------------------------------------------')

# Then we show the results
#print(predictList)

accuracy = np.where(pred==1)[0].shape[0]/pred.shape[0]
print('Accuracy: ', accuracy)

x0 = np.asarray(x0)
#x0 = [x0] * 3
#x0 = expand_img_dim(x0,1)
x0 = torch.from_numpy(x0).type('torch.FloatTensor')

print(x0.shape)
#imshow(torchvision.utils.make_grid(x0))
print('Ground truth: ', dataPath)

siftImg0 = imgArtCropper(cv2.imread(dataPath,0))
siftImg0 = cv2.resize(siftImg0, dim, interpolation = cv2.INTER_AREA)
siftImg0 = cv2.equalizeHist(siftImg0)

output1,_,_ = net(Variable(x0).cuda(),Variable(x0).cuda(),Variable(x0).cuda())


rankList = []
# Iterating through all activation maps
for key,value in featureMapDict.items():
  absPath,output2 = value
  euclidean_distance = F.pairwise_distance(output1, output2)
  #rankList.append((trueFilePath.split('/')[-2],euclidean_distance.item(),trueFilePath))
  rankList.append((key,euclidean_distance.item(),absPath,None))


rankList.sort(key = lambda x: x[1])

rankListFinal = []
for name,score,absPath,_ in rankList[:siftNLim]:
  #img1 = imgArtCropper(cv2.resize(cv2.imread(absPath,0), dim, interpolation = cv2.INTER_AREA))/255.0
  img1 = imgArtCropper(cv2.imread(absPath,0))
  img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

  numPoints = calculateHOGpoints(orb,siftImg0,img1)
  finalScore = calculateSIFTscore(score, numPoints)
  rankListFinal.append((name,finalScore,absPath,numPoints,score))


rankListFinal.sort(key = lambda x: x[1])



rankIter = 0
for name,score,absPath,numHOGPoints,originalScore in rankListFinal:
  print(absPath)


