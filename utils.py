# helper functions
import os
import random
import numpy as np
import torch
# import ivtmetrics
from sklearn.metrics import average_precision_score
import neptune.new as neptune
from torchmetrics import AveragePrecision as AP


def seed_torch(seed=42):
    """
    Seed various random number generators to ensure reproducibility.

    Args:
        seed (int): Seed value to set for random number generators.

    Returns:
        None
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Table for printing results

header = f"""
 Epoch | {"Loss":^6} | {"Val Loss":^7} | {"mAP":^7} | {"CmAP":^7} | {"Time, m":^6}
"""

raw_line = "{:6d} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:6.2f}"


def logging_to_neptune(CFG):
    """
    Log hyperparameters to Neptune for experiment tracking.

    Args:
        CFG (config object): Configuration object containing hyperparameters.

    Returns:
        neptune.run.Run: Neptune run object for tracking.
    """

    # Initiate logging
    run = neptune.init(
        project=CFG.neptune_project,
        api_token=CFG.neptune_api_token,
    )

    # Log hyperparameters to Neptune
    run["Model"].log(CFG.model_name)
    run["imsize"].log(CFG.height)
    run["LR"].log(CFG.lr)
    run["bs"].log(CFG.batch_size)
    run["Epochs"].log(CFG.epochs)
    run["SD"].log(CFG.SD)
    run["T_0"].log(CFG.T_0)
    run["min_lr"].log(CFG.min_lr)
    run["seed"].log(str(CFG.seed))
    run["split"].log(CFG.challenge_split)
    run["tsize"].log(CFG.target_size)
    run["smooth"].log(CFG.smooth)
    run["exp"].log(CFG.exp)

    return run


# def cholect45_ivtmetrics_mAP(df, CFG):
#     """
#     Compute the official CholecT45 mAP score.

#     Takes a dataframe with ground truth triplets and predictions.

#     Metric calculation:
#     - Aggregate per video over each fold
#     - Mean of 5 folds

#     Parameters:
#     - df (pd.DataFrame): DataFrame with ground truth triplets and predictions.
#     - CFG (object): Configuration object containing hyperparameters.

#     Returns:
#     float: Mean mAP value over 5 folds.

#     """
#     # Get the indexes of the 1st triplet/prediction columns
#     tri0_idx = int(df.columns.get_loc("tri0"))
#     pred0_idx = int(df.columns.get_loc("0"))

#     # Initiate empty list to store the folds mAP
#     ivt = []

#     # Loop over the 5 folds
#     for fold in range(CFG.n_fold):
#         # Initialize the ivt metric
#         rec = ivtmetrics.Recognition(num_class=100)

#         # Filter the fold and its corresponding videos
#         fold_df = df[df["fold"] == fold]
#         vids = fold_df.video.unique()

#         # Loop over the videos
#         for i, v in enumerate(vids):
#             # Filter the video
#             vid_df = fold_df[fold_df["video"] == v]

#             rec.update(
#                 vid_df.iloc[:, tri0_idx : tri0_idx + 100].values,
#                 vid_df.iloc[:, pred0_idx : pred0_idx + 100].values,
#             )

#             rec.video_end()

#         # Get the final mAP score for the fold
#         ivt.append(rec.compute_video_AP("ivt")["mAP"])

#     # Return the mean mAP value over 5 folds
#     return np.mean(ivt)

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

    """
    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(df.columns.get_loc("tri0"))
    pred0_idx = int(df.columns.get_loc("0"))

    # Initiate empty list to store the folds mAP
    ivt = []

    # Loop over the 5 folds
    for fold in range(CFG.n_fold):

        # Filter the fold and its corresponding videos
        fold_df = df[df["fold"] == fold]
        vids = fold_df.video.unique()

        
        # Compute using the ivtmetrics aggregation
        fold_mAP = per_epoch_ivtmetrics(fold_df, CFG)

        # Get the final mAP score for the fold
        ivt.append(fold_mAP)

    # Return the mean mAP value over 5 folds
    return np.mean(ivt)


# def per_epoch_ivtmetrics(fold_df, CFG):
#     """
#     Compute per-epoch ivtmetrics.

#     Parameters:
#     - fold_df (pd.DataFrame): DataFrame with ground truth triplets and predictions for a fold.
#     - CFG (object): Configuration object containing hyperparameters.

#     Returns:
#     float: mAP score for the given fold.

#     Example:
#     ```python
#     epoch_mAP = per_epoch_ivtmetrics(fold_df, CFG)
#     ```

#     """
#     # Get the indexes of the 1st triplet/prediction columns
#     tri0_idx = int(fold_df.columns.get_loc("tri0"))
#     pred0_idx = int(fold_df.columns.get_loc("0"))

#     # Initialize the ivt metric
#     rec = ivtmetrics.Recognition(num_class=100)

#     # Get unique videos
#     vids = fold_df.video.unique()

#     # Loop over the videos
#     for i, v in enumerate(vids):
#         # Filter the video
#         vid_df = fold_df[fold_df["video"] == v]

#         rec.update(
#             vid_df.iloc[:, tri0_idx : tri0_idx + 100].values,
#             vid_df.iloc[:, pred0_idx : pred0_idx + 100].values,
#         )

#         rec.video_end()

#     # Get the final mAP score
#     mAP = rec.compute_video_AP("ivt")["mAP"]

#     return mAP

def per_epoch_ivtmetrics(fold_df, CFG):

    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(fold_df.columns.get_loc(CFG.col0))
    pred0_idx = int(fold_df.columns.get_loc("0"))

    # Empty list to stack the [100 predictions] of each video
    all_classwise_stack = []

    # Get the fold's corresponding videos
    vids = fold_df.video.unique()

    # Loop over the videos
    for i, v in enumerate(vids):

        # Filter the video
        vid_df = fold_df[fold_df["video"] == v]

        torch_ap = AP(
            task="multilabel", num_labels=CFG.metric_tsize, average="none"
        ).to(CFG.device)

        # Metric
        classwise = torch_ap(
            torch.tensor(
                vid_df.iloc[:, pred0_idx : pred0_idx + CFG.metric_tsize].values
            ).to(CFG.device),
            torch.tensor(vid_df.iloc[:, tri0_idx : tri0_idx + CFG.metric_tsize].values)
            .long()
            .to(CFG.device),
        )

        # Append the [100 predictions] of each video
        all_classwise_stack.append(classwise.cpu())

    ### Calculate the mean of each category: [100 predictions]
    all_scores_fold = []

    # Loop over the Predictions of each video
    # print(len(all_classwise_stack))
    for j in range(len(all_classwise_stack[0])):  # (j: 0 -> 100)

        # Filter the j element of the [100 predictions] of each video
        xclass = [all_classwise_stack[vidx][j] for vidx in range(len(vids))]

        # Convert -0.0 to NaN
        xclass_NaN = [np.nan if x == -0.0 else x for x in xclass]

        # Mean of the triplet in the videos
        xclass_mean = np.nanmean(np.array(xclass_NaN))

        # Append the triplet score to the list until we have all the 100 triplets
        all_scores_fold.append(xclass_mean)

    # print(f"Fold mAP score: {np.nanmean(xclass_mean)}")

    # Mean of the 100 triplets
    overall_mAP = np.nanmean(np.array(all_scores_fold))


    return overall_mAP



def compute_mAP_score(valid_folds):
    """
    Compute mean Average Precision (mAP) score with no aggregation.

    Args:
        valid_folds (DataFrame): DataFrame containing predictions.

    Returns:
        float: The mean Average Precision (mAP) score.
    """

    # Score metrics for the fold
    tri0_idx = int(valid_folds.columns.get_loc("tri0"))
    pred0_idx = int(valid_folds.columns.get_loc("0"))

    # Compute the metric using sklearn
    classwise = average_precision_score(
        valid_folds.iloc[:, tri0_idx : tri0_idx + 100].values,
        valid_folds.iloc[:, pred0_idx : pred0_idx + 100].values,
        average=None,
    )

    # The mean mAP of all the (available) classes
    mAP = np.nanmean(classwise)
    return mAP


def print_training_info(folds, CFG):

    # print GPU model
    print("\033[94mHardware used\033[0m")
    print(f"GPU: {torch.cuda.get_device_name(0)}, cpu cores: {os.cpu_count()}\n")

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
        f"{'Target size:':<20} {CFG.target_size}\n"
        f"{'Self-distillation:':<20} {CFG.distill}\n"
        f"{'N° images used is:':<20} {len(folds)}\n"
        f"{'Experiment:':<20} {tag}\n"
    )

    # Print the training info
    print("\033[94mTraining parameters\033[0m")
    print(training_info)

    # Print the hyperparam info
    hyperparameters_info = (
        f"{'Starting LR:':<20} {CFG.lr}\n"
        f"{'Minimum LR:':<20} {CFG.min_lr}\n"
        f"{'Epochs:':<20} {CFG.epochs}\n"
        f"{'Batch size:':<20} {CFG.batch_size}\n"
    )

    print("\033[94mHyperparameters\033[0m")
    print(hyperparameters_info)

    metrics_info = (
        f"{'mAP:':<20} Overall mAP per fold (no aggregation)\n"
        f"{'cmAP:':<20} Challenge official mAP (aggregation per video)\n"
    )

    print("\033[94mMetrics\033[0m")
    print(metrics_info)

    print("\033[94mTraining started\033[0m\n")


def update_model_config(CFG):
    """
    Update the model configuration based on the specified experiment.

    Args:
        CFG (config object): Configuration object to be updated.

    Returns:
        None
    """

    if "MultiT" in CFG.exp:
        CFG.target_size = 131
    elif "phase" in CFG.exp:
        CFG.target_size = 138
    elif "Large" in CFG.exp:
        CFG.model_name = "swin_large_patch4_window7_224"
        CFG.target_size = 131
    else:
        CFG.target_size = 100


def save_predictions(CFG, pred_df):
    """
    Save predictions to a CSV file based on the specified configuration.

    Args:
        CFG (config object): Configuration object containing model settings.
        pred_df (DataFrame): DataFrame containing predictions.

    Returns:
        None
    """

    if CFG.inference:

        print("\033[94mSaving...\033[0m")

        if CFG.save_folder:
            folder = CFG.save_folder
            os.makedirs(os.path.join(CFG.output_dir, CFG.save_folder), exist_ok=True)

        else:
            folder = "predictions"

        # Save path for pretrained models and custom models
        if CFG.pretrained_model:
            preds_save_path = os.path.join(
                CFG.output_dir,
                f"{folder}/{CFG.exp}.csv",
            )
        else:
            preds_save_path = os.path.join(
                CFG.output_dir,
                f"{folder}/{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.csv",
            )

        pred_df.to_csv(preds_save_path)
        print(f"Predictions saved at {preds_save_path}")
