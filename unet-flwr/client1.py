import os
import sys

import time
import random

import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

from IPython.display import clear_output

# working with images
import cv2

import torchvision.transforms as transforms

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torchsummary
from tqdm import notebook

from utils.metrics import iou_pytorch_eval, iou_pytorch_test, IoULoss, IoUBCELoss
from utils.dataset import myDataSet
from unet import UNet
import flwr as fl

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

random_seed=42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # select device for training, i.e. gpu or cpu
print(DEVICE)

def train(model, dataloader_train, epochs):
    """Train the model on the training set."""
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
        print("\r Epoch: {} of {}, Iter.: {} of {}, Loss: {:.6f}".format(1, epochs, i, len(dataloader_train), running_loss/(i+1)), end="")
        
        running_iou += iou_pytorch_eval(prediction, masks)
        print("\r Epoch: {} of {}, Iter.: {} of {}, IoU:  {:.6f}".format(1, epochs, i, len(dataloader_train), running_iou/(i+1)), end="")

def test(model, dataloader_val):
    """Validate the model on the test set."""
    model.eval()
    val_loss = 0
    val_iou = 0
    for i, (imgs, masks) in enumerate(dataloader_val):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        
        prediction = model(imgs)
        loss = criterion(prediction, masks)
        val_loss += loss.item()
        
        val_iou += iou_pytorch_eval(prediction, masks)
       
    return val_loss, val_iou

def load_data():
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

    return dataloader_train, dataloader_val

# #############################################################################
# Federating the pipeline with Flower
# #############################################################################


# Load model and data (simple CNN, CIFAR-10)

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

model = UNet(channel_in=3,channel_out= 1)
model = model.to(DEVICE) # load model to DEVICE

# Define variables for the training
epochs = 1
patience=10

# Define optimiser and criterion for the training. You can try different ones to see which works best for your data and task
optimiser = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-8)

# criterion = nn.BCEWithLogitsLoss()
# model_name = 'UNet_BCEWithLogitsLoss_augmented'

criterion = IoULoss()
model_name = 'UNet_IoULoss_augmented'
trainloader, testloader = load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  def get_parameters(self):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

  def set_parameters(self, parameters):
    print("loaded parameters")
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):
    self.set_parameters(parameters)
    print("Training started")
    train(model, trainloader, epochs=1)
    return self.get_parameters(), len(trainloader.dataset), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    loss, accuracy = test(model, testloader)
    return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

# Start Flower client
fl.client.start_numpy_client("[::]:8080", client=FlowerClient())
