import time
import numpy as np
from torch.cuda import amp
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader

from augmentation import get_transforms
from dataset import TrainDataset

# Helper functions
class AverageMeter(object):
    def __init__(self):
        """
        Initialize AverageMeter attributes.
        """
        self.reset()

    def reset(self):
        """
        Reset the meter to its initial state.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Parameters:
        - val (float): Current value to be added to the running sum.
        - n (int): Number of occurrences of the value (default is 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(
    train_loader, model, CFG, criterion, optimizer, epoch, scheduler, device, scaler
):
    """
    Training loop function: loops over the dataloader.

    Parameters:
    - train_loader (DataLoader): DataLoader for training data.
    - model (nn.Module): PyTorch model to be trained.
    - CFG (Namespace): Configuration object containing hyperparameters.
    - criterion (nn.Module): Loss function for training.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - epoch (int): Current epoch number.
    - scheduler: Learning rate scheduler.
    - device (torch.device): Device (GPU or CPU) on which the training is performed.
    - scaler (torch.cuda.amp.GradScaler): PyTorch AMP scaler for mixed precision training.

    Returns:
    float: Average loss per epoch.
    """

    # Start variables
    losses = AverageMeter()
    global_step = 0

    # Switch to train mode
    model.train()

    for step, data in enumerate(train_loader):
        # Get the batch of images and labels
        images, labels = data
        batch_size = labels.size(0)

        # Start the optimizer
        optimizer.zero_grad()

        # Send the images and labels to the GPU
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

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            # Perform optimization step only after accumulating gradients for a specified number of steps
            scaler.step(optimizer)
            global_step += 1
            scaler.update()

    return losses.avg


def valid_fn(valid_loader, model, CFG, criterion, device):
    """
    Validation loop over the validation DataLoader.

    Parameters:
    - valid_loader (DataLoader): DataLoader for validation data.
    - model (Module): PyTorch model to be evaluated.
    - CFG (object): Configuration object containing hyperparameters.
    - criterion (Module): Loss function for validation.
    - device (object): Device (GPU or CPU) on which the validation is performed.

    Returns:
    tuple: Average loss, predictions (numpy array).

    """
    losses = AverageMeter()

    # Switch to evaluation mode
    model.eval()

    # Start a list to store the predictions
    preds = []

    # Loop over the DataLoader
    for step, data in enumerate(valid_loader):
        # Get the images and labels
        images, labels = data
        batch_size = labels.size(0)

        # Send images and labels to GPU
        images, labels = images.to(device), labels.to(device)

        # Eval mode
        with torch.no_grad():
            # Run the model on the validation set
            y_preds = model(images)

        # Compute the validation loss on the triplets only
        loss = criterion(y_preds[:, :100], labels[:, :100])

        # Update the loss
        losses.update(loss.item(), batch_size)

        # Update predictions
        preds.append(y_preds.sigmoid().to("cpu").numpy())

    # Concatenate predictions
    predictions = np.concatenate(preds)

    if CFG.gradient_accumulation_steps > 1:
        loss = loss / CFG.gradient_accumulation_steps

    return losses.avg, predictions


def inference_fn(
    valid_loader,
    model,
    device,
):
    """
    Inference loop over the validation DataLoader.

    Parameters:
    - valid_loader (DataLoader): DataLoader for inference data.
    - model (Module): PyTorch model for making predictions.
    - device (object): Device (GPU or CPU) on which the inference is performed.

    Returns:
    np.ndarray: Predictions as a NumPy array.

    """
    # Switch to evaluation mode
    model.eval()

    # Start a list to store the predictions
    preds = []

    # Loop over the DataLoader
    for step, images in enumerate(valid_loader):
        # Measure data loading time
        images = images.to(device)

        # Compute predictions
        with torch.no_grad():
            y_preds = model(images)

        preds.append(y_preds.sigmoid().to("cpu").numpy())

    # Concatenate predictions
    predictions = np.concatenate(preds)
    return predictions


def apply_self_distillation(fold, train_folds, CFG):
    """
    Apply self-distillation to the student model.

    Parameters:
    - fold: Current fold index.
    - train_folds: Training folds DataFrame.
    - CFG: Configuration object.

    Returns:
    pd.DataFrame: Updated training folds DataFrame after applying self-distillation.
    """
    # Read soft labels
    soft_labels_path = os.path.join(
        CFG.parent_path,
        f"softlabels/sl_f{fold}_{CFG.model_name[:8]}_{CFG.target_size}.csv",
    )
    train_softs = pd.read_csv(soft_labels_path)

    # Get the index of triplet 0 and soft label 0
    tri0_idx = train_folds.columns.get_loc("tri0")
    sl_pred0_idx = train_softs.columns.get_loc("0")

    # Reorder train soft labels to match the train labels order
    train_softs = train_softs.merge(train_folds[["image_id"]], on="image_id", how="right")

    # Apply self-distillation: Default SD=1
    tri_range = slice(tri0_idx, tri0_idx + CFG.target_size)
    sl_range = slice(sl_pred0_idx, sl_pred0_idx + CFG.target_size)
    train_folds.iloc[:, tri_range] = (
        train_folds.iloc[:, tri_range].values * (1 - CFG.SD)
        + train_softs.iloc[:, sl_range].values * CFG.SD
    )

    print("Soft-labels loaded successfully!")

    # Apply label smoothing
    if CFG.smooth:
        train_folds.iloc[:, tri_range] = (
            train_folds.iloc[:, tri_range] * (1.0 - CFG.ls) + 0.5 * CFG.ls
        )
    return train_folds


def get_dataloaders(train_folds, valid_folds, CFG):
    """
    Get PyTorch dataloaders for training and validation datasets.

    Parameters:
    - train_folds (pd.DataFrame): DataFrame containing training data.
    - valid_folds (pd.DataFrame): DataFrame containing validation data.
    - CFG (object): Configuration object containing hyperparameters.

    Returns:
    - DataLoader: PyTorch DataLoader for the training dataset.
    - DataLoader: PyTorch DataLoader for the validation dataset.

    """
    # PyTorch datasets
    # Apply train augmentations
    train_dataset = TrainDataset(
        train_folds, CFG, transform=get_transforms(data="train", CFG=CFG)
    )

    # Apply validation augmentations
    valid_dataset = TrainDataset(
        valid_folds, CFG, transform=get_transforms(data="valid", CFG=CFG)
    )

    # PyTorch train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.nworkers,
        pin_memory=True,
        drop_last=True,
    )

    # PyTorch valid dataloader
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.nworkers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, valid_loader


import pandas as pd


def get_dataframes(folds, fold):
    """
    Split the provided DataFrame into train and validation sets based on the given fold index.

    Parameters:
    - folds (pd.DataFrame): DataFrame containing the data with a "fold" column for splitting.
    - fold (int): Fold index used for validation set, while the rest are used for training.

    Returns:
    - train_folds (pd.DataFrame): DataFrame for the training set.
    - valid_folds (pd.DataFrame): DataFrame for the validation set.
    - temp (pd.DataFrame): Temporary DataFrame for metric computation.

    """
    # Get train and valid indexes
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    # Get train dataset
    train_folds = folds.loc[trn_idx].reset_index(drop=True)

    # Get valid dataset
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    # Temporary df to compute the metric
    temp = folds.loc[val_idx].reset_index(drop=True)

    return train_folds, valid_folds, temp
