import torch.utils.data
import torch.nn as nn
import torchvision
from PIL import Image
from pycocotools.coco import COCO
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate
from utils import create_pretrained_faster_Rcnn_model, collate_fn, get_transform, CustomCocoDataset
import config
from coco_utils import get_coco, get_coco_kp


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, config.num_classes),
        "coco_kp": (data_path, get_coco_kp, config.num_classes)
    }
    p, ds_fn, num_classes = paths[name]
    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


if __name__ == '__main__':
    dataset, num_classes = get_dataset(config.dataset_detection_type, "train",
                                       get_transform(train=True), config.root_data_dir)
    dataset_test, _ = get_dataset(config.dataset_detection_type, "val",
                                  get_transform(train=False), config.root_data_dir)

    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if config.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=config.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, config.train_batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, config.train_batch_size, drop_last=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=config.num_workers,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=config.num_workers,
        collate_fn=collate_fn)

    faster_rcnn_model = create_pretrained_faster_Rcnn_model(config.num_classes)
    faster_rcnn_model.to(device)
    print(faster_rcnn_model)

    params = [p for p in faster_rcnn_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=7,
                                                gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    best_acc = 0.0

    for epoch in range(config.num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(faster_rcnn_model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluator = evaluate(faster_rcnn_model, data_loader_test, device=device)

    # For inference
    faster_rcnn_model.eval()
    # predictions = faster_rcnn_model(x)

