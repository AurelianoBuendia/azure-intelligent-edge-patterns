import math
import sys
import copy
import time
import datetime
import torch
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
import copy


def train_model(model, device, data_loader, _optimizer):
    start_time = time.time()
    print("Start time: {}".format(datetime.timedelta(seconds=start_time)))
    result = model.train()
    i = 0
    running_loss = 0.0
    for imgs, annotations in data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        _optimizer.zero_grad()
        loss_dict = model(imgs, annotations)
        print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        _optimizer.step()
        running_loss += losses.item() * imgs.size(0)
    len_dataloader = len(data_loader)
    calculate_loss('train', len_dataloader, running_loss)
    end_time = time.time()
    print("End time: {}".format(datetime.timedelta(seconds=end_time)))
    print("Elapsed time: {} seconds".format(str(datetime.timedelta(seconds=end_time - start_time))))


def calculate_loss(phase, length, running_loss):
    epoch_loss = running_loss / length
    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    return epoch_loss


def calculate_accuracy(phase, length, running_corrects):
    epoch_acc = running_corrects.double() / length
    print('Acc: {:.4f}'.format(phase, epoch_acc))
    return epoch_acc


@torch.no_grad()
def test_model(model, device, data_loader, optimizer):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    best_model_wts = copy.deepcopy(model.state_dict())
    global best_acc
    for inputs, annotations in data_loader:
        optimizer.zero_grad()
        with torch.no_grad():  # Do not calculate gradients in test phase
            imgs = list(img.to(device) for img in inputs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item() * inputs.size(0)
    len_dataloader = len(data_loader)
    torch.cuda.synchronize()
    model_time = time.time()
    outputs = model(imgs)
    epoch_acc = calculate_accuracy('test', len_dataloader, running_corrects)
    # Store the weights of that model which has the best accuracy so far
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    return best_model_wts


def build_model(model, _data_loader, _optimizer, _scheduler, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        train_model(model, _data_loader, _optimizer)
        _scheduler.step()
        best_model_wts = test_model(model, _optimizer)
        print()
    print('Best test accuracy: {:4f}'.format(best_acc))
    # At the end of training load the model with best accuracy in test
    model.load_state_dict(best_model_wts)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


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


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    #coco = data_loader.dataset.coco
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
