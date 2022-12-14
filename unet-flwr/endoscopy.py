"""unet-augmented.ipynb


Automatically generated by Colaboratory.


Original file is located at

    https://colab.research.google.com/drive/1DDUTWMEmOJ8YyOVHOIBsC8tA8YUR_Dhx


## Step1: Put all libraries and packages at top


### Standard imports

"""


# files and system

import os

import sys


import time

import random


import numpy as np

import matplotlib.pyplot as plt


from IPython.display import clear_output


# working with images

import cv2

import imageio




import torchvision.transforms as transforms


import torch

import torchvision

import torch.nn as nn

import torch.nn.functional as F


import torchsummary

from tqdm import notebook


"""### Custom imports"""


# losses

from utils.metrics import iou_pytorch_eval, iou_pytorch_test, IoULoss, IoUBCELoss


# dataset

from utils.dataset import myDataSet


sys.path.insert(0, './notebooks')


sys.path.insert(0, '../models')


# models

from unet import UNet


sys.path.insert(0, '../notebooks')


"""## Step 2: Fix a seed for reproducibility, and choose the DEVICE"""


random_seed=42

random.seed(random_seed)

np.random.seed(random_seed)

torch.manual_seed(random_seed)

torch.cuda.manual_seed(random_seed)

torch.backends.cudnn.deterministic = True


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # select device for training, i.e. gpu or cpu

print(DEVICE)




"""## Step 3: Set your transforms, e.g, normalisation, resizing, rotation, flip, padding etc"""


_size = 256, 256

resize = transforms.Resize(_size, interpolation=0)


# set your transforms 

train_transforms = transforms.Compose([

                           transforms.Resize(_size, interpolation=0),

                           transforms.RandomRotation(180), # allow any rotation

                           transforms.RandomHorizontalFlip(0.5),

                           transforms.RandomCrop(_size, padding = 10), # needed after rotation (with original size)

                       ])


test_transforms = transforms.Compose([

                           transforms.Resize(_size, interpolation=0),

                       ])


"""## Step 4: Make your train and validation data loader with option to augment or not"""


path_images = "./data/train/images"

path_masks = "./data/train/masks"


train_images_filenames = list(sorted(os.listdir(os.path.join("./data/train/", "images"))))

val_images_filenames = list(sorted(os.listdir(os.path.join("./data/val/", "images"))))



custom_dataset_train = myDataSet(train_images_filenames, path_images, path_masks, transforms=train_transforms)

custom_dataset_val = myDataSet(val_images_filenames, path_images, path_masks, transforms=test_transforms)


print("My custom training-dataset has {} elements".format(len(custom_dataset_train)))

print("My custom validation-dataset has {} elements".format(len(custom_dataset_val)))


BATCH_SIZE = 2


# Create dataloaders from datasets with the native pytorch functions

dataloader_train = torch.utils.data.DataLoader(custom_dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

dataloader_val = torch.utils.data.DataLoader(custom_dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)



"""# Training hyperparameters"""


# Begin training

model = UNet(channel_in=3, channel_out=1)

model = model.to(DEVICE) # load model to DEVICE


# Define variables for the training

epochs = 10

patience=10


# Define optimiser and criterion for the training. You can try different ones to see which works best for your data and task

optimiser = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-8)


# criterion = nn.BCEWithLogitsLoss()

# model_name = 'UNet_BCEWithLogitsLoss_augmented'


criterion = IoULoss()

model_name = 'UNet_IoULoss_augmented'


# criterion = IoUBCELoss()

# model_name = 'UNet_IoUBCELoss_augmented'


"""# Training"""




# Commented out IPython magic to ensure Python compatibility.

# %%time

# 


train_losses = []

val_losses = []

best_iou = 0

best_loss = np.Inf

best_epoch = -1

state = {}



for epoch in range(epochs):

    running_loss = 0

    running_iou = 0

    # Train

    model.train()

    for i, (imgs, masks) in enumerate(dataloader_train):

        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        

        prediction = model(imgs)

        

        optimiser.zero_grad()

        loss = criterion(prediction, masks)

        loss.backward()

        optimiser.step()

        

        running_loss += loss.item()

        print("\r Epoch: {} of {}, Iter.: {} of {}, Loss: {:.6f}".format(epoch, epochs, i, len(dataloader_train), running_loss/(i+1)), end="")

        

        running_iou += iou_pytorch_eval(prediction, masks)

        print("\r Epoch: {} of {}, Iter.: {} of {}, IoU:  {:.6f}".format(epoch, epochs, i, len(dataloader_train), running_iou/(i+1)), end="")

        

    # Validate

    model.eval()

    val_loss = 0

    val_iou = 0

    for i, (imgs, masks) in enumerate(dataloader_val):

        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        

        prediction = model(imgs)

        loss = criterion(prediction, masks)

        val_loss += loss.item()

        print("\r Epoch: {} of {}, Iter.: {} of {}, Loss: {:.6f}, Val. Loss: {:.6f}".format(epoch, epochs, len(dataloader_train), len(dataloader_train), running_loss/len(dataloader_train), val_loss/(i+1)), end="")

        

        val_iou += iou_pytorch_eval(prediction, masks)

        print("\r Epoch: {} of {}, Iter.: {} of {}, IoU: {:.6f}, Val. IoU: {:.6f}".format(epoch, epochs, len(dataloader_train), len(dataloader_train), running_iou/len(dataloader_train), val_iou/(i+1)), end="")

    

    

    # compute overall epoch losses

    epoch_train_loss = running_loss/len(dataloader_train)

    train_losses.append(epoch_train_loss)

    epoch_val_loss = val_loss/len(dataloader_val)

    val_losses.append(epoch_val_loss)

    # compute overall epoch iou-s

    epoch_train_iou = running_iou/len(dataloader_train)

    epoch_val_iou = val_iou/len(dataloader_val)

    

    print("\r Epoch: {} of {}, Iter.: {} of {}, Train Loss: {:.6f}, IoU: {:.6f}".format(epoch, epochs, len(dataloader_train), len(dataloader_train), epoch_train_loss, epoch_train_iou))

    print("\r Epoch: {} of {}, Iter.: {} of {}, Valid Loss: {:.6f}, IoU: {:.6f}".format(epoch, epochs, len(dataloader_train), len(dataloader_train), epoch_val_loss, epoch_val_iou))
