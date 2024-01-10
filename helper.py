import time
import numpy as np
from torch.cuda import amp
import torch
from tri_index import index_by_occurrence

from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)
import torch.nn as nn


# Helper functions
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(
    train_loader, model, CFG, criterion, optimizer, epoch, scheduler, device, scaler
):
    losses = AverageMeter()

    # switch to train mode
    model.train()

    model.train()
    for step, data in enumerate(train_loader):

        # Get the batch of images and labels
        images, labels = data
        batch_size = labels.size(0)

        # Start the optimizer
        optimizer.zero_grad()

        # Send the images and labels to gpu
        images = images.to(device)
        labels = labels.to(device)

        # Apply mixed precision
        with amp.autocast():

            # Get the predictions
            y_preds = model(images)

            # Compute the loss on multitask or triplets only
            if CFG.multi:
                loss = criterion(y_preds, labels)
            else:
                loss = criterion(y_preds[:, :100], labels[:, :100])

        # Update the loss
        losses.update(loss.item(), batch_size)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def valid_fn(valid_loader, model, CFG, criterion, device):

    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluation mode
    model.eval()

    # Start a list to store the predictions
    preds = []

    # Loop over the dataloader
    for step, data in enumerate(valid_loader):

        # Get the images and labels
        images, labels = data
        batch_size = labels.size(0)

        # Send images and labels to gpu
        images = images.to(device)
        labels = labels.to(device)

        # Eval mode
        with torch.no_grad():

            # Run the model on the validation set
            y_preds = model(images)

        # Compute the validation loss on the triplets only
        loss = criterion(y_preds[:, :100], labels[:, :100])

        # Update the loss
        losses.update(loss.item(), batch_size)

        # Update predictions
        preds.append(y_preds.to("cpu").numpy())

    # Concat and predictions
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def inference_fn(valid_loader, model, device):
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    for step, images in enumerate(valid_loader):

        # measure data loading time
        images = images.to(device)

        # compute loss
        with torch.no_grad():

            y_preds = model(images)

        preds.append(y_preds.sigmoid().to("cpu").numpy())

    predictions = np.concatenate(preds)
    return predictions
