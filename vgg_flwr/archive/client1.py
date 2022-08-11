from collections import OrderedDict
import warnings
from pathlib import Path
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
from endoscopy_nets import ClassificationModel
from endoscopy_data_splitter import EndoscopyDataSplitter
from endoscopy_dataset import EndoscopyDataset
# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

train_file_name = os.path.join("./endoscopy_splits", "site-1" + ".pkl")
valid_file_name = os.path.join("./endoscopy_valid", "valid.pkl")

device = "cpu"

def train(model, train_loader, epochs):
    """Train the model on the training set."""
    optimizer = optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training and Testing

    for epoch in range(epochs):
        model.train()
        epoch_len = len(train_loader)
        correct, total = 0, 1
        
        avg_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).type(torch.float))
           
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            print("epoch ", epoch, ",", " loss: ", loss.item()," Accuracy : ", (correct / total *100),"%")
        
def test(model, valid_loader):
    """Validate the model on the test set."""
    optimizer = optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.eval()
    metric = 0.0
    loss = 0.0
    with torch.no_grad():
        correct, total = 0, 0
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred_labels = torch.round(torch.sigmoid(outputs)).cpu().numpy()
            loss += criterion(outputs, labels.unsqueeze(1).type(torch.float))
            total += inputs.data.size()[0]
            match = np.array(pred_labels == labels.unsqueeze(1).type(torch.float).cpu().numpy())
            correct += match.sum().item()
        metric = correct / float(total) #Accuracy
    
    return loss / len(valid_loader.dataset), metric

def load_data():
    """Load CIFAR-10 (training and test set)."""
    if os.path.exists("./endoscopy_valid/valid.pkl") is False:
        splitter = EndoscopyDataSplitter(num_sites=1, alpha=0)
        splitter.split()

    train_dataset = EndoscopyDataset(pkl_path=train_file_name)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    valid_dataset = EndoscopyDataset(pkl_path=valid_file_name)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)

    return train_loader, valid_loader

# #############################################################################
# Federating the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = ClassificationModel().to(device)
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
fl.client.start_numpy_client("127.0.0.1:8080", client=FlowerClient(),root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),)
