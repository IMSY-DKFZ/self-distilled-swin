import pandas as pd
import os
from sklearn.model_selection import GroupKFold


def get_folds(n_fold, CFG):

    print("\033[94mPreparing the data\033[0m")

    # Read the dataframe
    train = pd.read_csv(os.path.join(CFG.parent_path, CFG.path_csv))

    print(f"Preprocessing the data...")

    # Get the list of videos
    vids = list(train.video.unique())

    # Start a folds df to map the folds
    folds = train.copy()

    # Official cross validation split of the CholecT45 dataset
    if CFG.challenge_split:

        fold1 = [
            "VID79",
            "VID02",
            "VID51",
            "VID06",
            "VID25",
            "VID14",
            "VID66",
            "VID23",
            "VID50",
        ]
        fold2 = [
            "VID80",
            "VID32",
            "VID05",
            "VID15",
            "VID40",
            "VID47",
            "VID26",
            "VID48",
            "VID70",
        ]
        fold3 = [
            "VID31",
            "VID57",
            "VID36",
            "VID18",
            "VID52",
            "VID68",
            "VID10",
            "VID08",
            "VID73",
        ]
        fold4 = [
            "VID42",
            "VID29",
            "VID60",
            "VID27",
            "VID65",
            "VID75",
            "VID22",
            "VID49",
            "VID12",
        ]
        fold5 = [
            "VID78",
            "VID43",
            "VID62",
            "VID35",
            "VID74",
            "VID01",
            "VID56",
            "VID04",
            "VID13",
        ]

        # Initiate the fold column
        folds["fold"] = -1

        # Map the folds to the videos
        fold_list = [fold1, fold2, fold3, fold4, fold5]
        for n, valfold in enumerate(fold_list):

            # Loop over the dataset to map the videos to their fold number
            for j, i in enumerate(folds.video.values):
                if i in valfold:
                    folds.loc[j, "fold"] = n

        folds["fold"] = folds["fold"].astype(int)

    # Using groupKFold split instead of the official split
    else:
        # Use groupKFold to split the videos
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
