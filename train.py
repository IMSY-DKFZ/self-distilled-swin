# Empty dataframe to store oofs and metrics
import gc
import os
import time
import torch
import pandas as pd
import neptune.new as neptune
import ivtmetrics
from sklearn.metrics import average_precision_score
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda import amp


from models import TripletModel
from preprocess import get_folds
from augmentation import *
from utils import *
from helper import *
from tri_index import *


def train_fnt(CFG):

    # Start timer
    start_time = time.time()

    # Seed everything for reproducibility
    seed_torch(seed=CFG.seed)

    # DEBUG: Faster iteration
    if CFG.debug:
        CFG.epochs = 1
        CFG.neplog = False

    # Log the hyperparams to neptune
    if CFG.neplog:

        # Initiate logging
        run = neptune.init(
            project=CFG.neptune_project,
            api_token=CFG.neptune_api_token,
        )  # your credential

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

    # Start an empty dataframe to store the predictions
    oof_df = pd.DataFrame()

    # Create folders to save the checkpoints and predictions
    os.makedirs(os.path.join(CFG.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(CFG.output_dir, "oofs"), exist_ok=True)

    # Get the preprocessed train dataframe
    folds, vids = get_folds(CFG.n_fold, CFG)

    print_training_info(folds, CFG)

    # List to store mAP results
    mAP_folds = []

    # Loop over the folds
    for fold in range(CFG.n_fold):

        # Skip some folds
        if fold in CFG.trn_fold:

            print("\033[92m" + f"{'-' * 8} Fold {fold + 1} / {CFG.n_fold}" + "\033[0m")

            # Load model and send it to the GPU
            model = TripletModel(CFG, CFG.model_name, pretrained=CFG.pretrained).to(CFG.device)

            # Get train and valid Dataframes
            train_folds, valid_folds, valid_folds_temp = get_dataframes(folds, fold)

            # Apply self-distillation to the train dataset
            if CFG.distill:
                train_folds = apply_self_distillation(fold, train_folds, CFG)

            # Get dataloders
            train_loader, valid_loader = get_dataloaders(train_folds, valid_folds, CFG)

            # Optimizer, scheduler and criterion
            optimizer = Adam(
                model.parameters(),
                lr=CFG.lr,
                weight_decay=CFG.weight_decay,
                amsgrad=False,
            )

            # Cosine annealing scheduler
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=CFG.T_0,
                T_mult=1,
                eta_min=CFG.min_lr,
                last_epoch=-1,
            )

            # Binary cross entropy loss function
            criterion = nn.BCEWithLogitsLoss(reduction="sum").to(CFG.device)

            # Set the variables to calculate and save the stats
            best_score = 0.0

            # Mixed precision scaler
            scaler = amp.GradScaler()

            # Score metrics for the fold
            tri0_idx = int(valid_folds.columns.get_loc("tri0"))

            # Start training: Loop over epochs
            print(header)
            for epoch in range(CFG.epochs):

                # Start epoch timer
                epoch_start = time.time()

                # TRAINING LOOP
                avg_loss = train_fn(
                    train_loader,
                    model,
                    CFG,
                    criterion,
                    optimizer,
                    epoch,
                    scheduler,
                    CFG.device,
                    scaler=scaler,
                )

                # VALIDATION LOOP
                avg_val_loss, preds = valid_fn(
                    valid_loader, model, CFG, criterion, CFG.device
                )

                # Update scheduler
                scheduler.step()

                # Get the updated lr after the update
                cur_lr = scheduler.get_last_lr()

                # original mAP from sklearn
                classwise = average_precision_score(
                    valid_folds.iloc[:, tri0_idx : tri0_idx + 100].values,
                    preds[:, :100],
                    average=None,
                )

                # In case of newer sklearn versions
                classwise[classwise == 0] = np.nan

                # Mean of all the available triplets
                mAP = np.nanmean(classwise)

                # Store the predictions in a temp df
                valid_folds_temp[[str(c) for c in range(CFG.target_size)]] = preds

                # ivtmetrics mAP score [Per video aggregation]
                cholect45_epoch_CV = per_epoch_ivtmetrics(valid_folds_temp, CFG)

                # Log the epoch results to neptune
                if CFG.neplog:
                    run[f"tloss{fold}"].log(avg_loss)
                    run[f"val_loss{fold}"].log(avg_val_loss)
                    run[f"mAP_0_{fold}"].log(mAP)
                    run[f"cmAP{fold}"].log(cholect45_epoch_CV)
                    run[f"cLR_{fold}"].log(cur_lr)

                # Print loss/metric
                print(
                    raw_line.format(
                        epoch,
                        avg_loss,
                        avg_val_loss,
                        mAP,
                        cholect45_epoch_CV,
                        (time.time() - epoch_start) / 60 ** 1,
                    )
                )

                # Save checkpoints
                save_checkpoint_path = os.path.join(
                    CFG.output_dir,
                    f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.pth",
                )

                # Save only best model (best mAP score)
                if mAP > best_score:
                    best_score = mAP
                    torch.save(
                        {"model": model.state_dict(), "preds": preds},
                        save_checkpoint_path,
                    )

            # Load the best checkpoint to calculate the fold's final score
            check_point = torch.load(save_checkpoint_path)

            # Save the best predictions to dataframe
            valid_folds[[str(c) for c in range(CFG.target_size)]] = check_point["preds"]

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
            fold_mAP = np.nanmean(classwise)

            # CholecT45 official metric
            cholect45_fold_CV = per_epoch_ivtmetrics(valid_folds, CFG)

            # Store the per fold mAP scores
            mAP_folds.append(cholect45_fold_CV)

            # Print fold metrics
            fold_header = f"{'=' * 20} Fold {fold} {'=' * 20}"
            fold_footer = "=" * len(fold_header)

            print(fold_header)
            print(f"  mAP: {fold_mAP:.4f}")
            print(f"  CholecT45 mAP: {cholect45_fold_CV:.4f}")
            print(fold_footer)

            if CFG.neplog:
                run[f"CV Folds"].log(fold_mAP)

            # Save predictions
            oof_df = pd.concat([oof_df, valid_folds])

            # Free the GPU memory
            del model, train_loader, valid_loader, train_folds, valid_folds
            torch.cuda.empty_cache()
            gc.collect()

    # Calculate overall mAP metric (No aggregation)
    classwise = average_precision_score(
        oof_df.iloc[:, tri0_idx : tri0_idx + 100].values,
        oof_df.iloc[:, pred0_idx : pred0_idx + 100].values,
        average=None,
    )
    overall_mAP = np.nanmean(classwise)

    # Get the final cross-validation score based on ivtmetrics
    cholect45_final_CV = cholect45_ivtmetrics_mAP(oof_df, CFG)

    # Print final metrics
    print(
        f"CV: OVERALL SCORES\n"
        f"  Overall mAP: {overall_mAP:.4f}\n"
        f"  CholecT45 mAP: {cholect45_final_CV:.4f}"
    )

    if CFG.neplog:
        run["CV"].log(overall_mAP)
        run["CholecT45_mAP"].log(cholect45_final_CV)

    # Save predictions
    save_path = f"{CFG.output_dir}/oofs/O_{CFG.model_name[:8]}_{CFG.exp}.csv"

    if not CFG.debug:
        oof_df.to_csv(save_path)

    if CFG.neplog:
        run[CFG.exp].upload(save_path)

    # Training time
    print(f"Training time: {(time.time() - start_time) / 60}")
