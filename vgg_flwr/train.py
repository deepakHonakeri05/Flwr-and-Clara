import torch
import torchvision
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

train_file_name = os.path.join("./endoscopy_splits", "site-1" + ".pkl")
valid_file_name = os.path.join("./endoscopy_valid", "valid.pkl")

if os.path.exists("./endoscopy_valid/valid.pkl") is False:
    splitter = EndoscopyDataSplitter(num_sites=1, alpha=0)
    splitter.split()

train_dataset = EndoscopyDataset(pkl_path=train_file_name)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

valid_dataset = EndoscopyDataset(pkl_path=valid_file_name)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 3

if __name__ == '__main__':

    print("Number of train samples: ", len(train_dataset))

    model = ClassificationModel().to(device)
    #model.load_state_dict(torch.load("cmcd_model_e10.pth"))

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

        
    model.eval()
    metric = 0.0
    with torch.no_grad():
        correct, total = 0, 0
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred_labels = torch.round(torch.sigmoid(outputs)).cpu().numpy()
            total += inputs.data.size()[0]
            match = np.array(pred_labels == labels.unsqueeze(1).type(torch.float).cpu().numpy())
            correct += match.sum().item()
        metric = correct / float(total)
            
    print(metric)


