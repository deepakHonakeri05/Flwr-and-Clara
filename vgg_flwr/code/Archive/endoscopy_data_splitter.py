import os
import cv2
import json
import pickle
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
from itertools import islice
from torchvision import models, transforms
from nvflare.apis.fl_context import FLContext
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent

ENDOSCOPY_DATA_SOURCE = '/tmp/endoscopy_data/'
ENDOSCOPY_VALID_DESTINATION = '/tmp/endoscopy_valid/'
ENDOSCOPY_SPLITS_DESTINATION = '/tmp/endoscopy_splits/'


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        self.pooling = model.avgpool
        self.flatten = nn.Flatten()
        self.fc = model.classifier[0]

    def forward(self, x):
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(512),
    transforms.Resize(448),
    transforms.ToTensor()
])


def _get_site_class_summary(site_idx):
    class_sum = dict()
    for site, data_idx in site_idx.items():
        class_sum[site] = len(data_idx)
    return class_sum


class EndoscopyDataSplitter(FLComponent):
    def __init__(self,
                 split_dir: str = ENDOSCOPY_SPLITS_DESTINATION,
                 num_sites: int = 8,
                 alpha: float = 0.5,
                 seed: int = 0
                 ):
        super().__init__()
        self.split_dir = split_dir
        self.valid_dir = ENDOSCOPY_VALID_DESTINATION
        self.num_sites = num_sites
        self.alpha = alpha
        self.seed = seed
        np.random.seed(self.seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = FeatureExtractor(models.vgg16(pretrained=True)).to(self.device)

        normal_images = list(os.listdir(os.path.join(ENDOSCOPY_DATA_SOURCE, "normal")))
        abnormal_images = list(os.listdir(os.path.join(ENDOSCOPY_DATA_SOURCE, "abnormal")))
        random.shuffle(normal_images)
        random.shuffle(abnormal_images)

        total_normal_n = len(normal_images)
        total_abnormal_n = len(abnormal_images)
        valid_normal_n = int(0.1 * total_normal_n)
        valid_abnormal_n = int(0.1 * total_abnormal_n)
        train_normal_n = len(normal_images[valid_normal_n:])
        train_abnormal_n = len(abnormal_images[valid_abnormal_n:])

        self.X_valid = list()
        self.y_valid = list()
        for i in range(valid_normal_n):
            self.X_valid.append(os.path.join(ENDOSCOPY_DATA_SOURCE, "normal/" +
                                             normal_images[i]))
            self.y_valid.append(1)
        for i in range(valid_abnormal_n):
            self.X_valid.append(os.path.join(ENDOSCOPY_DATA_SOURCE, "abnormal/" +
                                             abnormal_images[i]))
            self.y_valid.append(0)

        self.X_train = list()
        self.y_train = list()
        for i in range(train_normal_n):
            self.X_train.append(os.path.join(ENDOSCOPY_DATA_SOURCE, "normal/" +
                                             normal_images[valid_normal_n + i]))
            self.y_train.append(1)
        for i in range(train_abnormal_n):
            self.X_train.append(os.path.join(ENDOSCOPY_DATA_SOURCE, "abnormal/" +
                                             abnormal_images[valid_abnormal_n + i]))
            self.y_train.append(0)

        self.train_n = len(self.X_train)
        self.valid_n = len(self.X_valid)
        self.train_idxs = [i for i in range(self.train_n)]
        self.valid_idxs = [i for i in range(self.valid_n)]
        random.shuffle(self.train_idxs)
        random.shuffle(self.valid_idxs)

        if os.path.exists(ENDOSCOPY_SPLITS_DESTINATION):
            shutil.rmtree(ENDOSCOPY_SPLITS_DESTINATION)
        if os.path.exists(ENDOSCOPY_VALID_DESTINATION):
            shutil.rmtree(ENDOSCOPY_VALID_DESTINATION)

        if alpha < 0.:
            raise ValueError(f"Alpha should be larger 0.0 but was {alpha}!")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.split(fl_ctx)

    def split(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Partition ENDOSCOPY dataset into {self.num_sites} "
                              f"sites with Dirichlet sampling under alpha {self.alpha}")
        site_idx, class_sum = self._partition_data()

        if self.split_dir is None:
            raise ValueError("You need to define a valid `split_dir` when splitting the data.")
        if not os.path.isdir(self.split_dir):
            os.makedirs(self.split_dir)
        sum_file_name = os.path.join(self.split_dir, "summary.txt")
        with open(sum_file_name, "w") as sum_file:
            sum_file.write(f"Number of clients: {self.num_sites} \n")
            sum_file.write(f"Dirichlet sampling parameter: {self.alpha} \n")
            sum_file.write("Class counts for each client: \n")
            sum_file.write(json.dumps(class_sum))

        site_file_path = os.path.join(self.split_dir, "site-")
        for site in range(self.num_sites):
            train_data = list()
            site_file_name = site_file_path + str(site + 1) + ".pkl"
            for idx in site_idx.get(site):
                image_path = self.X_train[idx]
                im_array = cv2.imread(image_path)
                img = transform(im_array)
                img = img.reshape(1, 3, 448, 448)
                img = self.model(img).cpu().detach().numpy().reshape(-1)
                label = self.y_train[idx]
                train_data.append((img, label))
            with open(site_file_name, 'wb') as f:
                pickle.dump(train_data, f)

        if self.valid_dir is None:
            raise ValueError("You need to define a valid `valid_dir` when preparing the validation data.")
        if not os.path.isdir(self.valid_dir):
            os.makedirs(self.valid_dir)
        valid_data = list()
        valid_file_name = os.path.join(self.valid_dir, "valid.pkl")
        for idx in self.valid_idxs:
            image_path = self.X_valid[idx]
            im_array = cv2.imread(image_path)
            img = transform(im_array)
            img = img.reshape(1, 3, 448, 448)
            img = self.model(img).cpu().detach().numpy().reshape(-1)
            label = self.y_valid[idx]
            valid_data.append((img, label))
        with open(valid_file_name, 'wb') as f:
            pickle.dump(valid_data, f)

    def _partition_data(self):
        site_idx = dict()
        if self.alpha > 0:
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_sites))
            length_to_split = [int(self.train_n * proportion) for proportion in proportions]
            if sum(length_to_split) < self.train_n:
                d = self.train_n - sum(length_to_split)
                length_to_split[-1] = length_to_split[-1] + d
        else:
            length_to_split = [self.train_n]
        idxs_iter = iter(self.train_idxs)
        idxs_split = [list(islice(idxs_iter, elem)) for elem in length_to_split]
        for i in range(self.num_sites):
            site_idx[i] = idxs_split[i]
        class_sum = _get_site_class_summary(site_idx)
        return site_idx, class_sum
