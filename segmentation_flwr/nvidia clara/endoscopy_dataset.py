import torch
import pickle
import numpy as np
import torch.utils.data


class EndoscopyDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_path, transforms=None):
        self.pkl_path = pkl_path
        self.transforms = transforms
        with open(self.pkl_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        img = self.data[idx][0]
        mask = self.data[idx][1]
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        masks = mask == obj_ids[:, None, None]
        boxes = list()
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        is_crowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = is_crowd
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.data)
