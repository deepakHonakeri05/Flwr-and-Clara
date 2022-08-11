import os
import torch
import math
import torchvision
from torchvision import transforms as T

import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from endoscop_nets import SegmentationModel
from endoscopy_dataset import EndoscopyDataset
from utils import reduce_dict


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def get_transform(train):
    transforms = list()
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class ENDOSCOPYLearner:
    def __init__(
        self,
        train_idx_root: str = '../endoscopy_splits',
        aggregation_epochs: int = 1,
        lr: float = 1e-2,
        batch_size: int = 2,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_idx_root = train_idx_root
        self.valid_idx_root = '../endoscopy_test'
        self.aggregation_epochs = aggregation_epochs
        self.lr = lr
        self.best_acc = 0.0
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.writer = None

        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0

        # following will be created in initialize() or later
        self.app_root = None
        self.client_id = None
        self.local_model_file = None
        self.best_local_model_file = None
        self.writer = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.criterion_prox = None
        self.transform_train = None
        self.transform_valid = None
        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.client_id = 'site-1'

    def initialize(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SegmentationModel().to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.transform_train = None
        self.transform_valid = None
        self._create_datasets()

    def _create_datasets(self):
        if self.train_dataset is None or self.train_loader is None:
            train_file_name = os.path.join(self.train_idx_root, self.client_id + ".pkl")
            self.train_dataset = EndoscopyDataset(pkl_path=train_file_name, transforms=get_transform(train=True))
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                collate_fn=utils.collate_fn)

        if self.valid_dataset is None or self.valid_loader is None:
            valid_file_name = os.path.join(self.valid_idx_root, "valid.pkl")
            self.valid_dataset = EndoscopyDataset(pkl_path=valid_file_name, transforms=get_transform(train=False))
            self.valid_loader = torch.utils.data.DataLoader(
                self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                collate_fn=utils.collate_fn
            )

    def local_train(self, val_freq: int = 0):
        scaler = None
        for epoch in range(self.aggregation_epochs):
            self.model.train()
            epoch_len = len(self.train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            print(f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})")
            avg_loss = 0.0

            lr_scheduler = None
            if epoch == 0:
                warmup_factor = 1.0 / 1000
                warmup_iters = min(1000, len(self.train_loader) - 1)

                lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
                )
            for i, (images, targets) in enumerate(self.train_loader):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    print(loss_dict_reduced)
                    return

                self.optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(losses).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    losses.backward()
                    self.optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()

                current_step = epoch_len * self.epoch_global + i
                avg_loss = avg_loss + loss_value
                break

            acc = self.local_valid(self.valid_loader, tb_id="val_acc_local_model")
            if acc > self.best_acc:
                self.best_acc = acc
                self.save_model(is_best=True)

    def save_model(self, is_best=False):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
        if is_best:
            save_dict.update({"best_acc": self.best_acc})
            torch.save(save_dict, self.best_local_model_file)
        else:
            torch.save(save_dict, self.local_model_file)

    def local_valid(self, valid_loader, tb_id=None):
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.model.eval()

        coco = get_coco_api_from_dataset(valid_loader.dataset)
        iou_types = _get_iou_types(self.model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for i, (images, targets) in enumerate(valid_loader):
            images = list(img.to(self.device) for img in images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            outputs = self.model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)
            break
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        metric = coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        if tb_id:
            self.writer.add_scalar(tb_id, metric, self.epoch_global)
        return metric
