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


class dataset_voc(Dataset):
  def __init__(self, root_dir, trvaltest, transform=None):
      self.root_dir = root_dir
      self.DataGetter = DataGetter(self.root_dir)
      self.transform = transform
      if trvaltest == 0:
          self.imgfilenames, self.labels = self.DataGetter.trainingSettWithLabels()
      elif trvaltest == 1:
          self.imgfilenames, self.labels = self.DataGetter.valSettWithLabels()
      elif trvaltest == 2:
          self.imgfilenames, self.labels = self.DataGetter.testSettWithLabels()
      else:
          print("Invalid dataset classification in second positional argument, please try: 0, 1 or 2 if test data is availible")

      self.labels = torch.from_numpy(self.labels)

  def __len__(self):
      return len(self.imgfilenames)

  def __getitem__(self, idx):
    image = self.transform(PIL.Image.open(os.path.join(self.DataGetter.img_dir, self.imgfilenames[idx]+".jpg")))
    label = self.labels[idx]

    sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}

    return sample



def train_epoch(model,  trainloader,  criterion, device, optimizer ):

    #TODO
    #model.train() or model.eval() ?
    model.train()

    losses = []
    for batch_idx, data in enumerate(trainloader):
      #TODO
      inputs = data['image'].to(device)
      labels = data['label'].to(device)
      optimizer.zero_grad()

      #forward
      with torch.set_grad_enabled(True):
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          losses.append(loss)

          #backward
          loss.backward()
          optimizer.step()

    return np.mean(losses)


def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):

    model.eval()

    #curcount = 0
    #accuracy = 0

    concat_pred=[np.empty(shape=((len(dataloader.dataset)))) for _ in range(numcl)] #prediction scores for each class. each numpy array is a list of scores. one score per image; Changed the array size
    concat_labels=[np.empty(shape=((len(dataloader.dataset)))) for _ in range(numcl)] #labels scores for each class. each numpy array is a list of labels. one label per image; Changed the array size
    avgprecs=np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader

    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):


          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ',batch_idx)

          inputs = data['image'].to(device)
          outputs = model(inputs)

          labels = data['label']

          loss = criterion(outputs, labels.to(device) )
          losses.append(loss.item())

          #this was an accuracy computation
          #cpuout= outputs.to('cpu')
          #_, preds = torch.max(cpuout, 1)
          #labels = labels.float()
          #corrects = torch.sum(preds == labels.data)
          #accuracy = accuracy*( curcount/ float(curcount+labels.shape[0]) ) + corrects.float()* ( curcount/ float(curcount+labels.shape[0]) )
          #curcount+= labels.shape[0]

          #TODO: collect scores, labels, filenames
          fnames.append(data['filename'])
          predictions = torch.round(outputs)

          for clIndex in range(numcl):
              for i in range(inputs.shape[0]): #should be batch size
                  concat_pred[clIndex][batch_idx+i] = outputs[i][clIndex]
                  #concat_labels[clIndex][batch_idx+i] = predictions[i][clIndex] was not sure wich labels you wanted
                  concat_labels[clIndex][batch_idx+i] = labels[i][clIndex]

    #calculating mean average precision
    thresholds = np.arange(start=0.3, stop=0.7, step=0.05) #thresholds evenly spaced around 0.5
    for c in range(numcl):
        precisions = []
        recalls = []

        for thr in thresholds:

            temp_pred = concat_pred[c]
            for i in range(len(temp_pred)):
                if temp_pred[i] >= thr:
                    temp_pred[i] = 1
                else:
                    temp_pred[i] = 0

            precision = sklearn.metrics.precision_score(y_true=concat_labels[c], y_pred=temp_pred)
            recall = aklearn.metrics.recall_core(y_true=concat_labels[c], y_pred=temp_pred)
            precisions.append(precision)
            recalls.append(recall)
        precisions.append(1)
        recalls.append(0)
        precision = np.array(precisions)
        recalls = np.array(recalls)

        avgprecs[c]= numpy.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])

    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames


def traineval2_model_nocv(dataloader_train, dataloader_test ,  model ,  criterion, optimizer, scheduler, num_epochs, device, numcl):

  best_measure = 0
  best_epoch =-1

  trainlosses=[]
  testlosses=[]
  testperfs=[]

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    avgloss=train_epoch(model,  dataloader_train,  criterion,  device , optimizer )
    trainlosses.append(avgloss)

    if scheduler is not None:
      scheduler.step()

    perfmeasure, testloss,concat_labels, concat_pred, fnames  = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)

    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)

    avgperfmeasure = np.mean(perfmeasure)
    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)

    if avgperfmeasure > best_measure: #higher is better or lower is better?
      bestweights = model.state_dict()

      #TODO track current best performance measure and epoch
      best_measure = avgperfmeasure
      best_epoch = epoch

      #TODO save your scores
      with open("file_names.txt", "w") as nameFile:
          for name in fnames:
              nameFile.write(name)

      with open("ground_truth.txt", "w") as file:
          concat_labels = np.array(concat_labels)
          for i in range(concat_labels.shape[1]):
              file.write(concat_labels[:,i])

      with open("prediction_scores.txt", "w") as file:
          concat_pred = np.array(concat_pred)
          for i in range(concat_pred.shape[1]):
              file.write(concat_pred[:,i])

      with open("predicted_labels.txt", "w") as file:
          concat_pred = np.array(concat_pred)
          for i in range(concat_pred.shape[1]):
              file.write(np.round(concat_pred[:,i]))


  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs




class yourloss(nn.modules.loss._Loss):

    def __init__(self, reduction: str = 'mean') -> None:
        super(yourloss, self).__init__()
        self.lossfunc = nn.BCEWithLogitsLoss()

    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        loss = self.lossfunc(input_, target)
        return loss

def runstuff():


  config = dict()

  config['use_gpu'] = False #True #TODO change this to True for training on the cluster, eh
  config['lr']=0.005
  config['batchsize_train'] = 16
  config['batchsize_val'] = 64
  config['maxnumepochs'] = 2

  config['scheduler_stepsize']=10
  config['scheduler_factor']=0.3



  # kind of a dataset property
  config['numcl']=20



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


  #datasets
  image_datasets={}
  image_datasets['train']=dataset_voc(root_dir="../../../../VOCdatasett/VOC2012",trvaltest=0, transform=data_transforms['train'])
  image_datasets['val']=dataset_voc(root_dir="../../../../VOCdatasett/VOC2012",trvaltest=1, transform=data_transforms['val'])

  #dataloaders
  #TODO use num_workers=1
  dataloaders = {}
  dataloaders['train'] = DataLoader(dataset=image_datasets["train"], batch_size=config['batchsize_train'], num_workers=1)
  dataloaders['val'] = DataLoader(dataset=image_datasets["val"], batch_size=config['batchsize_val'], num_workers=1)


  #device
  if True == config['use_gpu']:
      device= torch.device('cuda:0')

  else:
      device= torch.device('cpu')

  #model
  #TODO
  model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
  #overwrite last linear layer
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, 20)

  model = model.to(device)

  lossfct = yourloss()

  #TODO
  # Observe that all parameters are being optimized
  someoptimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])

  # Decay LR by a factor of 0.3 every X epochs
  #TODO
  num_epochs = config['maxnumepochs']
  X = round(num_epochs/7.0)
  if X < 1:
      X = 1

  milestones = list(range(X-1, num_epochs, X))
  somelr_scheduler = torch.optim.lr_scheduler.MultiStepLR(someoptimizer, milestones, gamma = 0.3)

  best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'] )

  print("Best epoch: {}".format(best_epoch))
  print("Best measure: {}".format(best_measure))

###########
# for part2
###########

if __name__=='__main__':

  runstuff()
