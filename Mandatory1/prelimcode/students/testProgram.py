import random
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt

from torch import Tensor

import time
import os
import numpy as np

import PIL.Image
import sklearn.metrics

from vocparseclslabels import PascalVOC
from vocDataGetter import DataGetter

from typing import Callable, Optional


from train_pytorch_in5400tmp_v1_studentversion import dataset_voc

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
model.fc.out_features = 20

numcl = 20
concat_pred=[np.empty(shape=(0)) for _ in range(numcl)]

#data augmentations
data_transforms = {
  'train': transforms.Compose([
      transforms.Resize(256),
      transforms.RandomCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ]),
  'val': transforms.Compose([
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ]),
}

trainingSett = dataset_voc("../../../../VOCdatasett/VOC2012", trvaltest=0, transform=data_transforms['train'])

valSett = dataset_voc("../../../../VOCdatasett/VOC2012", 1)

dataloaders = {}
dataloaders['train'] = DataLoader(dataset=trainingSett, batch_size=4, num_workers=1)

for batch_idx, data in enumerate(dataloaders['train']):
    inputs = data['image']
    print(len(inputs))
    print(len(data['label']))

"""
dataGetter = DataGetter("../../../../VOCdatasett/VOC2012")
categories = dataGetter.list_image_sets()

for i in range(12):
    #random_index = random.randint(0,len(trainingSett)-1)
    random_sample = trainingSett.__getitem__(i)
    image = random_sample["image"]
    plt.figure()
    plt.imshow(image)
    labels = ""
    for j in range(len(categories)):
        if random_sample["label"][j] == 1:
            labels = labels +" "+categories[j]

    plt.title("Random training sample, labels:"+labels)
    plt.show()

    sample = valSett[i]
    image = sample["image"]
    plt.figure()
    plt.imshow(image)
    labels = ""
    for j in range(len(categories)):
        if sample["label"][j] == 1:
            labels = labels +" "+categories[j]

    plt.title("Val sample, labels:"+labels)
    plt.show()
"""
