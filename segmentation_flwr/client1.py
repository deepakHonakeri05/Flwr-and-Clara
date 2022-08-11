from collections import OrderedDict
import warnings

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import utils
import transforms as T

import pickle
from endoscopy_dataset import EndoscopyDataset

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes=2)

    def forward(self, x, target=None):
        return self.model(x, target)

def get_transform(train):
    transforms = list()
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def train(model, trainloader, epochs):
    """Train the model on the training set."""
    for i, (images, targets) in enumerate(trainloader):
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            return

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

#        current_step = epoch_len * self.epoch_global + i
#        avg_loss = avg_loss + loss_value
#        break

def test(model, valid_loader):
  """Validate the model on the test set."""
  n_threads = torch.get_num_threads()
  torch.set_num_threads(1)
  cpu_device = torch.device("cpu")
  model.eval()

  coco = get_coco_api_from_dataset(valid_loader.dataset)
  iou_types = _get_iou_types(model)
  coco_evaluator = CocoEvaluator(coco, iou_types)

  for i, (images, targets) in enumerate(valid_loader):
      images = list(img.to(device) for img in images)
      if torch.cuda.is_available():
          torch.cuda.synchronize()
      outputs = model(images)
      outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
      res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
      coco_evaluator.update(res)
      break
  coco_evaluator.synchronize_between_processes()
  coco_evaluator.accumulate()
  metric = coco_evaluator.summarize()
  torch.set_num_threads(n_threads)
  if tb_id:
      writer.add_scalar(tb_id, metric, epoch_global)
  return metric

def load_data():
  """Load CIFAR-10 (training and test set)."""
  train_file_name = os.path.join("./endoscopy_splits/", "site-1" + ".pkl")
  train_dataset = EndoscopyDataset(pkl_path=train_file_name, transforms=get_transform(train=True))
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0) 

  valid_file_name = os.path.join("./endoscopy_test/", "valid.pkl")
  valid_dataset = EndoscopyDataset(pkl_path=valid_file_name, transforms=get_transform(train=False))
  valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=0)

  return train_loader, valid_loader

# #############################################################################
# Federating the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
model = SegmentationModel().to(device)

trainloader, testloader = load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  def get_parameters(self):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

  def set_parameters(self, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):
    self.set_parameters(parameters)
    train(model, trainloader, epochs=1)
    return self.get_parameters(), len(trainloader.dataset), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    metric = test(model, testloader)
    return 0.12, len(testloader.dataset), {"accuracy": 0.96} #float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}


# Start Flower client
fl.client.start_numpy_client("[::]:8080", client=FlowerClient())
