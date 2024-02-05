# Empty dataframe to store oofs and metrics
import gc
import os
import time
import torch
import pandas as pd
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
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

    # Logging training stats
    if CFG.neplog:
        run = logging_to_neptune(CFG)

    # Start an empty dataframe to store the predictions
    oof_df = pd.DataFrame()

    # Create folders to save the checkpoints and predictions
    os.makedirs(os.path.join(CFG.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(CFG.output_dir, "oofs"), exist_ok=True)

    # Get the preprocessed train dataframe
    folds, vids = get_folds(CFG)

    print_training_info(folds, CFG)

    # Loop over the folds
    for fold in range(CFG.n_fold):

        # Skip some folds
        if fold in CFG.trn_fold:

            print("\033[92m" + f"{'-' * 8} Fold {fold + 1} / {CFG.n_fold}" + "\033[0m")

            # Load model and send it to the GPU
            model = TripletModel(CFG, CFG.model_name, pretrained=CFG.pretrained).to(
                CFG.device
            )

            # Get train and valid Dataframes
            train_folds, valid_folds, valid_folds_temp = get_dataframes(folds, fold)

            # Apply self-distillation to the train dataset
            if CFG.distill:
                train_folds = apply_self_distillation(fold, train_folds, CFG)

            # Get dataloders
            train_loader, valid_loader = get_dataloaders(train_folds, valid_folds, CFG)

            # Get optimize, scheduler and loss function
            optimizer, scheduler, criterion = compile_model(CFG, model)

            # Set the variables to calculate and save the stats
            best_score = 0.0

            # Mixed precision scaler
            scaler = amp.GradScaler()

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

                # Store the predictions in a temp df
                valid_folds_temp[[str(c) for c in range(CFG.target_size)]] = preds

                # Compute overall mAP score (no aggregation)
                mAP = compute_mAP_score(valid_folds_temp)

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

                # Save last epoch
                if CFG.early_stopping:
                    torch.save(
                        {"model": model.state_dict(), "preds": preds},
                        save_checkpoint_path,
                    )

                else:
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

            # Compute overall mAP score (no aggregation)
            fold_mAP = compute_mAP_score(valid_folds)

            # CholecT45 official metric
            cholect45_fold_CV = per_epoch_ivtmetrics(valid_folds, CFG)

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

    # Overall mAP (no aggregation)
    overall_mAP = compute_mAP_score(oof_df)

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
