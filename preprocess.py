import os
import pandas as pd
from sklearn.model_selection import GroupKFold

def get_folds(CFG):
    """
    Prepare and split the data into folds for cross-validation.

    Args:
        n_fold (int): Number of folds for cross-validation.
        CFG (OmegaConf): Configuration object.

    Returns:
        folds (DataFrame): DataFrame with fold assignments.
        vids (list): List of unique video IDs.
    """

    print("\033[94mPreparing the data\033[0m")

    # Read the dataframe
    train = pd.read_csv(os.path.join(CFG.parent_path, CFG.path_csv))

    print(f"Preprocessing the data...")

    # Get the list of videos
    vids = list(train.video.unique())

    # Start a folds df to map the folds
    folds = train.copy()

    # Official cross-validation split of the CholecT45 dataset
    if CFG.challenge_split:

        fold_map = {
            "fold1": [
                "VID79", "VID02", "VID51", "VID06", "VID25", "VID14", "VID66", "VID23", "VID50"
            ],
            "fold2": [
                "VID80", "VID32", "VID05", "VID15", "VID40", "VID47", "VID26", "VID48", "VID70"
            ],
            "fold3": [
                "VID31", "VID57", "VID36", "VID18", "VID52", "VID68", "VID10", "VID08", "VID73"
            ],
            "fold4": [
                "VID42", "VID29", "VID60", "VID27", "VID65", "VID75", "VID22", "VID49", "VID12"
            ],
            "fold5": [
                "VID78", "VID43", "VID62", "VID35", "VID74", "VID01", "VID56", "VID04", "VID13"
            ]
        }

        # Initialize the 'fold' column with -1
        folds["fold"] = -1

        # Map the folds to the videos
        for fold, video_list in fold_map.items():
            folds.loc[folds["video"].isin(video_list), "fold"] = int(fold[-1]) - 1

        # Convert the 'fold' column to int
        folds["fold"] = folds["fold"].astype(int)

    # Using GroupKFold split instead of the official split
    else:
        # Use GroupKFold to split the videos
        Fold = GroupKFold(n_splits=CFG.nfold)

        # Get the videos
        groups = folds.folder

        # Group by video and stratify by target distribution
        for n, (train_index, val_index) in enumerate(
            Fold.split(folds, folds[CFG.target_col], groups)
        ):
            folds.loc[val_index, "fold"] = int(n)

        folds["fold"] = folds["fold"].astype(int)

    # Print the folds distribution
    if CFG.debug:
        print(folds.groupby(["fold", "folder"])["multi_tri"].count())

    print("Dataset ready!\n")

    return folds, vids
