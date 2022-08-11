import os
import json
import pickle
import random
import numpy as np
from PIL import Image
from itertools import islice


def _get_site_class_summary(site_idx):
    class_sum = dict()
    for site, data_idx in site_idx.items():
        class_sum[site] = len(data_idx)
    return class_sum


class EndoscopyDataSplitter:
    def __init__(self,
                 split_dir: str = '../endoscopy_splits',
                 num_sites: int = 2,
                 alpha: float = 0.5,
                 seed: int = 0
                 ):
        super().__init__()
        self.train_root = '../endoscopy_train'
        self.valid_root = '../endoscopy_test'
        self.split_dir = split_dir
        self.valid_dir = '../endoscopy_test'
        self.num_sites = num_sites
        self.alpha = alpha
        self.seed = seed
        np.random.seed(self.seed)
        self.train_images = list(sorted(os.listdir(os.path.join(self.train_root, "images"))))
        self.train_masks = list(sorted(os.listdir(os.path.join(self.train_root, "masks"))))
        self.valid_images = list(sorted(os.listdir(os.path.join(self.valid_root, "images"))))
        self.valid_masks = list(sorted(os.listdir(os.path.join(self.valid_root, "masks"))))
        self.train_n = len(self.train_images)
        self.valid_n = len(self.valid_images)
        self.train_idxs = [i for i in range(self.train_n)]
        self.valid_idxs = [i for i in range(self.valid_n)]
        random.shuffle(self.train_idxs)
        random.shuffle(self.valid_idxs)

        if alpha < 0.:
            raise ValueError(f"Alpha should be larger 0.0 but was {alpha}!")

    def handle_event(self):
        self.split()

    def split(self):
        print(f"Partition ENDOSCOPY dataset into {self.num_sites} sites with Dirichlet sampling under alpha {self.alpha}")
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
            for idx in self.train_idxs:
                image_path = os.path.join(self.train_root, "images/" + self.train_images[idx])
                mask_path = os.path.join(self.train_root, "masks/" + self.train_masks[idx])
                img = Image.open(image_path).convert("RGB")
                img = img.resize((512, 512))
                mask = Image.open(mask_path).convert('1')
                mask = mask.resize((512, 512))
                train_data.append((img, mask))
            with open(site_file_name, 'wb') as f:
                pickle.dump(train_data, f)

        if self.valid_dir is None:
            raise ValueError("You need to define a valid `valid_dir` when preparing the validation data.")
        if not os.path.isdir(self.valid_dir):
            os.makedirs(self.valid_dir)
        valid_data = list()
        valid_file_name = os.path.join(self.valid_dir, "valid.pkl")
        for idx in self.valid_idxs:
            image_path = os.path.join(self.valid_root, "images/" + self.valid_images[idx])
            mask_path = os.path.join(self.valid_root, "masks/" + self.valid_masks[idx])
            img = Image.open(image_path).convert("RGB")
            img = img.resize((512, 512))
            mask = Image.open(mask_path).convert('1')
            mask = mask.resize((512, 512))
            valid_data.append((img, mask))
        with open(valid_file_name, 'wb') as f:
            pickle.dump(valid_data, f)

    def _partition_data(self):
        site_idx = dict()
        proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_sites))
        length_to_split = [int(self.train_n * proportion) for proportion in proportions]
        if sum(length_to_split) < self.train_n:
            d = self.train_n - sum(length_to_split)
            length_to_split[-1] = length_to_split[-1] + d
        idxs_iter = iter(self.train_idxs)
        idxs_split = [list(islice(idxs_iter, elem)) for elem in length_to_split]
        for i in range(self.num_sites):
            site_idx[i] = idxs_split[i]
        class_sum = _get_site_class_summary(site_idx)
        return site_idx, class_sum
