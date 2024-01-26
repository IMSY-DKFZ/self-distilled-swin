# helper functions
import os
import random
import numpy as np
import torch
import ivtmetrics


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Table for printing results

header = f"""
 Epoch | {"Loss":^7} | {"Val Loss":^8} | {"mAP":^8} | {"CmAP":^8} | {"Time, m":^7}
"""

raw_line = "{:6d} | {:7.3f} | {:8.3f} | {:8.3f} | {:8.3f} | {:6.2f}"


def cholect45_ivtmetrics_mAP(df, CFG):
    """
    Compute the official CholecT45 mAP score.

    Takes a dataframe with ground truth triplets and predictions.

    Metric calculation:
    - Aggregate per video over each fold
    - Mean of 5 folds

    Parameters:
    - df (pd.DataFrame): DataFrame with ground truth triplets and predictions.
    - CFG (object): Configuration object containing hyperparameters.

    Returns:
    float: Mean mAP value over 5 folds.

    Example:
    ```python
    mean_mAP = CholecT45_ivtmetrics_mAP(df, CFG)
    ```

    """
    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(df.columns.get_loc("tri0"))
    pred0_idx = int(df.columns.get_loc("0"))

    # Initiate empty list to store the folds mAP
    ivt = []

    # Loop over the 5 folds
    for fold in range(CFG.n_fold):
        # Initialize the ivt metric
        rec = ivtmetrics.Recognition(num_class=100)

        # Filter the fold and its corresponding videos
        fold_df = df[df["fold"] == fold]
        vids = fold_df.video.unique()

        # Loop over the videos
        for i, v in enumerate(vids):
            # Filter the video
            vid_df = fold_df[fold_df["video"] == v]

            rec.update(
                vid_df.iloc[:, tri0_idx : tri0_idx + 100].values,
                vid_df.iloc[:, pred0_idx : pred0_idx + 100].values,
            )

            rec.video_end()

        # Get the final mAP score for the fold
        ivt.append(rec.compute_video_AP("ivt")["mAP"])

    # Return the mean mAP value over 5 folds
    return np.mean(ivt)


def per_epoch_ivtmetrics(fold_df, CFG):
    """
    Compute per-epoch ivtmetrics.

    Parameters:
    - fold_df (pd.DataFrame): DataFrame with ground truth triplets and predictions for a fold.
    - CFG (object): Configuration object containing hyperparameters.

    Returns:
    float: mAP score for the given fold.

    Example:
    ```python
    epoch_mAP = per_epoch_ivtmetrics(fold_df, CFG)
    ```

    """
    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(fold_df.columns.get_loc("tri0"))
    pred0_idx = int(fold_df.columns.get_loc("0"))

    # Initialize the ivt metric
    rec = ivtmetrics.Recognition(num_class=100)

    # Get unique videos
    vids = fold_df.video.unique()

    # Loop over the videos
    for i, v in enumerate(vids):
        # Filter the video
        vid_df = fold_df[fold_df["video"] == v]

        rec.update(
            vid_df.iloc[:, tri0_idx : tri0_idx + 100].values,
            vid_df.iloc[:, pred0_idx : pred0_idx + 100].values,
        )

        rec.video_end()

    # Get the final mAP score
    mAP = rec.compute_video_AP("ivt")["mAP"]

    return mAP


def print_training_info(folds, CFG):
    # Experiment tag
    tag = (
        f"\033[92m{CFG.exp}\033[0m"
        if CFG.exp != "myexp"
        else f"\033[91mPlease tag your experiment; i.e: exp=teacher_multitask\033[0m\n"
    )
    # Create a formatted training info string
    training_info = (
        f"{'Model:':<20} {CFG.model_name}\n"
        f"{'Multitask:':<20} {False if CFG.target_size==100 else True}\n"
        f"{'Self-distillation:':<20} {CFG.distill}\n"
        f"{'N° images used is:':<20} {len(folds)}\n"
        f"{'Experiment:':<20} {tag}\n"
    )

    # Print the formatted training info
    print("\033[94mTraining parameters\033[0m")
    print(training_info)

    # Print the formatted training info

    hyperparameters_info = (
        f"{'Starting LR:':<20} {CFG.lr}\n"
        f"{'Minimum LR:':<20} {CFG.min_lr}\n"
        f"{'Epochs:':<20} {CFG.epochs}\n"
        f"{'Batch size:':<20} {CFG.batch_size}\n"
    )

    print("\033[94mHyperparameters\033[0m")
    print(hyperparameters_info)

    # print GPU model
    print("\033[94mHardware used\033[0m")
    print(f"GPU: {torch.cuda.get_device_name(0)}, cpu cores: {os.cpu_count()}\n")

    print("\033[94mTraining started\033[0m\n")
