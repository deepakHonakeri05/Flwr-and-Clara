from collections import OrderedDict
import warnings

import flwr as fl
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from custom_network import CustomNetwork
from endoscopy_data_splitter import EndoscopyDataSplitter

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
TRAIN_DATA_PATH = "/Users/deepak/utd/QuEST/fed AI/vgg_flwr/dataset/"
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

def train(net, train_data_loader, epochs):
    """Train the model on the training set."""
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        correct, total = 0, 1

        for step, (x, y) in enumerate(train_data_loader):
            b_x = Variable(x)   # batch x (image)
            b_y = F.one_hot(y, num_classes=2)  # batch y (target)
            output = net(b_x)

#            print(output, "\n", b_y)
            loss = loss_func(output, b_y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#            print("epoch ", epoch, ",", "loss: ", loss.item())

            total += b_y.size(0)
            correct += (torch.max(output.data, 1)[1] == y).sum().item()
            print("epoch ", epoch, ",", " loss: ", loss.item()," Accuracy : ", (correct / total *100),"%")
        
def test(net, val_data_loader):
    """Validate the model on the test set."""
    correct, total, loss = 0, 1, 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()
    
    net.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(val_data_loader):
          output = net(Variable(x))
          yt = F.one_hot(y, num_classes=2)
          loss += loss_func(output, yt.float()).item()
          total += yt.size(0)
          correct += (torch.max(output.data, 1)[1] == y).sum().item()
    return loss / len(val_data_loader.dataset), correct / total

def load_data():
    """Load CIFAR-10 (training and test set)."""
    train_file_name = os.path.join(self.train_idx_root, self.client_id + ".pkl")
    valid_file_name = os.path.join(self.valid_idx_root, "valid.pkl")
    
    splitter = EndoscopyDataSplitter(num_sites=1, alpha=0)
    splitter.split(fl_ctx)
    
    
    self.train_dataset = EndoscopyDataset(pkl_path=train_file_name)
    self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    self.valid_dataset = EndoscopyDataset(pkl_path=valid_file_name)
    self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    return train_loader, valid_loader

# #############################################################################
# Federating the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = CustomNetwork()
trainloader, testloader = load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  def get_parameters(self):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

  def set_parameters(self, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):
    self.set_parameters(parameters)
    train(net, trainloader, epochs=1)
    return self.get_parameters(), len(trainloader.dataset), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    loss, accuracy = test(net, testloader)
    return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

# Start Flower client
fl.client.start_numpy_client("[::]:8080", client=FlowerClient())
