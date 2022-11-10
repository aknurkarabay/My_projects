from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import copy
import pandas as pd

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = 'food1k'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=False, num_workers=4)
              for x in ['train','val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:6") #if torch.cuda.is_available() else "cpu")
########################################
def visualize_model(model, num_images=1):
    #was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    df = pd.DataFrame(columns=['ground_truth', 'top1', 'top5'])

    with torch.no_grad():

        
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            #print(preds.cpu().detach().numpy(), labels.cpu().detach().numpy())
            gt = labels.cpu().detach().numpy()[0]
            pred = preds.cpu().detach().numpy()[0]
            df.at[i, 'ground_truth'] = gt

            predByClass = outputs.cpu().detach().numpy()
            predByClass = predByClass[0].tolist()
            top5 = set()
            
            for j in range(5):
                max_val = max(predByClass)
                max_ind = predByClass.index(max_val)
                top5.add(max_ind)
                predByClass[max_ind] = -10000
                
            if gt in top5:
                df.at[i, 'top5'] = 1
            else:
                df.at[i, 'top5'] = 0

            if gt == pred:
                df.at[i, 'top1'] = 1 
            else:
                df.at[i, 'top1'] = 0
    print(df.head(100))
    df.to_csv('accuracy_metrics_food1k_rs101.csv')
        
model = torch.load('food1k_rs101.pt')

visualize_model(model)

df = pd.read_csv('accuracy_metrics_food1k_rs101.csv')

print(df['ground_truth'].sum(), df['top1'].sum(),df['top5'].sum())

print(df['top1'].sum()/len(df), df['top5'].sum()/len(df))
