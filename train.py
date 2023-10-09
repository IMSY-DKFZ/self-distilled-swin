# Empty dataframe to store oofs and metrics
import gc
from multiprocessing import reduction
import os
import time
import torch
import pandas as pd
from models import *
from preprocess import get_folds
from tri_index import index_by_occurrence, videos_index

import ivtmetrics
from sklearn.metrics import average_precision_score

from losses import SmoothBCEwLogits
from dataset import *
from augmentation import *
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)
from utils import *
from helper import get_scheduler, train_fn, valid_fn
from torch.cuda import amp
from multihead import *
from multihead_phaseprediction import *
from hybridm import *

from temporal import *
from tri_index import *
from losses import *
from bitemp import *

import sklearn
import os
import glob
import neptune.new as neptune





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
        

    # Start an empty dataframe to store the predictions
    oof_df = pd.DataFrame()
    
    # Output directory to store the output: weights, predictions...
    output_dir = os.path.join(CFG.parent_path, "output/")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the preprocessed train dataframe
    folds, vids = get_folds(CFG.n_fold, CFG)


    # Print training info
    print(
        f"Model: {CFG.model_name}; MultiTask: {CFG.multi}; Self-distillation: {CFG.SD}; N° images used is:{len(folds)}; image size{CFG.height, CFG.width} \n {CFG.exp}"
    )
    
    # List to store mAP results
    mAP_folds = []


    # Loop over the folds
    for fold in range(CFG.n_fold):

        # Skip some folds
        if fold in CFG.trn_fold:
            print(f"Training fold {fold}")

            ##Load model and send it to the GPU
            model = TripletModel(CFG, CFG.model_name, pretrained=True).to(
                        CFG.device
                    )

            ##Get train and valid indexes
            trn_idx = folds[folds["fold"] != fold].index
            val_idx = folds[folds["fold"] == fold].index

            # Get train dataset
            train_folds = folds.loc[trn_idx].reset_index(drop=True)
           

            # Get valid dataset
            valid_folds = folds.loc[val_idx].reset_index(drop=True)
            valid_folds2 = folds.loc[val_idx].reset_index(drop=True)

           
           
            # Apply self-distillation to the student model only
            if CFG.do_SD:

                train_softs = pd.read_csv(
                        os.path.join(
                            CFG.parent_path,
                            f"dataframes/tsl_gsp0_{CFG.target_size}_{fold}.csv",
                        )
                    )

                # Get the index of triplet 0 and softlabel 0
                tri0_idx = int(folds.columns.get_loc("tri0"))
                sl_pred0_idx = int(train_softs.columns.get_loc("0"))

                # In case of independent test set
                if CFG.test_inference:
                    clips = ["VID68", "VID70", "VID73", "VID74", "VID75"]
                    soft_labels = soft_labels[
                        ~soft_labels.video.isin(clips)
                    ].reset_index(drop=True)

            
                ## Reorder train soft labels to match the train labels order
                train_softs = train_softs.merge(
                    train_folds[
                        [
                            "id2",
                        ]
                    ],
                    on="id2",
                    how="right",
                )

               
                # Apply self distillation: Default SDn= 1
                train_folds.iloc[:, tri0_idx : tri0_idx + CFG.target_size] = (
                    train_folds.iloc[:, tri0_idx : tri0_idx + CFG.target_size].values
                    * (1 - CFG.SD)
                    + train_softs.iloc[
                        :, sl_pred0_idx : sl_pred0_idx + CFG.target_size
                    ].values
                    * CFG.SD
                )

                # Apply label smoothing
                if CFG.msmooth:
                    train_folds.iloc[:, tri0_idx : tri0_idx + CFG.sd_size] = (
                        train_folds.iloc[:, tri0_idx : tri0_idx + CFG.sd_size]
                        * (1.0 - CFG.ls)
                        + 0.5 * CFG.ls
                    )

            
            # Pytorch datasets
            # Apply train augmentations
            train_dataset = TrainDataset(
                train_folds, CFG, transform=get_transforms(data="train", CFG=CFG)
            )
            
            # Apply validation augmentations
            valid_dataset = TrainDataset(
                valid_folds,
                CFG,
                transform=get_transforms(data="valid", CFG=CFG)            )

           
            # Pytorch train dataloader
            train_loader = DataLoader(
                train_dataset,
                batch_size=CFG.batch_size,
                shuffle=True,
                num_workers= CFG.workers,
                pin_memory=True,
                drop_last=True,
            )

            # Pytorch valid dataloader
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=CFG.batch_size ,
                shuffle=False,
                num_workers=nworkers,
                pin_memory=False,
                drop_last=False,
            )

            # Optimizer, scheduler and criterion
            optimizer = Adam(
                model.parameters(),
                lr=CFG.lr,
                weight_decay=CFG.weight_decay,
                amsgrad=False,
            )
            
            # Cosine annealing scheduler
            scheduler = CosineAnnealingWarmRestarts(
                        optimizer, T_0=CFG.epochs + 1, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
           
           
            # Binary cross entropy loss function
            criterion = nn.BCEWithLogitsLoss(reduction="sum").to(
                    CFG.device
                )  # reduction='sum'
         

            # Set the variables to calculate and save the stats
            best_score = 0.0

            # Mixed precision scaler
            scaler = amp.GradScaler()

            # Score metrics for the fold
            tri0_idx = int(valid_folds.columns.get_loc("tri0"))

            ##Initiate the ivt metric
            recognize = ivtmetrics.Recognition(num_class=100)

            # Start training: Loop over epochs
            print(header)
            for epoch in range(CFG.epochs):

                # Start epoch timer
                epoch_start = time.time()

                # Reset the metric at the begining of every epoch
                recognize.reset()

                ##TRAINING LOOP
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
                mAP = np.nanmean(classwise)

               
                # Log the epoch results to neptune
                if CFG.neplog:
                    run[f"tloss{fold}"].log(avg_loss)
                    run[f"val_loss{fold}"].log(avg_val_loss)
                    run[f"mAP_0_{fold}"].log(mAP)
                    run[f"imAP_0_{fold}"].log(0)
                    run[f"cLR_{fold}"].log(cur_lr)
                    run[f"cmAP_{fold}"].log(overall_mAP)

                
                # Print loss/metric
                print(
                    raw_line.format(
                        epoch,
                        avg_loss,
                        avg_val_loss,
                        mAP,
                        mAP,
                        (time.time() - epoch_start) / 60 ** 1,
                    )
                )

             
                # Save checkpoints
                save_checkpoint_path =  os.path.join(output_dir,
                 f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.epochs}_{CFG.SD}_{CFG.exp}.pth")
            
                # Save only best model (best mAP score)
                if mAP > best_score:
                    best_score = mAP
                    torch.save(
                        {"model": model.state_dict(), "preds": preds},
                        save_checkpoint_path,
                    )



            # Load the best checkpoint to calculate the fold's final score
            check_point = torch.load(
               save_checkpoint_path)
            
            
            # Save the best predictions to dataframe
            valid_folds[[str(c) for c in range(CFG.target_size)]] = check_point["preds"]

            # Score metrics for the fold
            tri0_idx = int(valid_folds.columns.get_loc("tri0"))
            pred0_idx = int(valid_folds.columns.get_loc("0"))

            
            # Compute fold's overall mAP score
            recognize.reset_global()
            recognize.update(
                valid_folds.iloc[:, tri0_idx : tri0_idx + 100].values,
                valid_folds.iloc[:, pred0_idx : pred0_idx + 100].values,
            )
           
            # Final results
            results_ivt = recognize.compute_AP("ivt")
            fold_mAP = results_ivt["mAP"]

            
            # Compute the metric using sklearn
            classwise = average_precision_score(
                valid_folds.iloc[:, tri0_idx : tri0_idx + 100].values,
                valid_folds.iloc[:, pred0_idx : pred0_idx + 100].values,
                average=None,
            )
            
            
            # The mean Map of all the classes
            mean = np.nanmean(classwise)

            # Store the per fold mAP scores
            mAP_folds.append(fold_mAP)


            # Print the fold's CV score
            print("---------------")
            print(f"Fold{fold}: \n mAP: {round(fold_mAP, 4)} \n mean: {round(mean, 4)}")
            print("---------------")

            if CFG.neplog:
                run[f"CV Folds"].log(mean)

            # Save OOFs + Metrics
            oof_df = pd.concat([oof_df, valid_folds])

            # Free the GPU memory
            del model, train_dataset, valid_dataset, train_loader, valid_loader
            torch.cuda.empty_cache()
            gc.collect()

    
    # Calculate the final metric score over all the folds
    recognize.reset_global()
    recognize.update(
        oof_df.iloc[:, tri0_idx : tri0_idx + 100].values,
        oof_df.iloc[:, pred0_idx : pred0_idx + 100].values,
    )
    results_ivt = recognize.compute_AP("ivt")
    CV_mAP = results_ivt["mAP"]

    classwise = average_precision_score(
        oof_df.iloc[:, tri0_idx : tri0_idx + 100].values,
        oof_df.iloc[:, pred0_idx : pred0_idx + 100].values,
        average=None,
    )
    mean = np.nanmean(classwise)

    mAP_folds_mean = np.array(mAP_folds).mean()

    print(
        f"CV: OVERALL:  \nmAP:{round(CV_mAP, 4)} \n mean:{round(mean, 4)} \n folds_mean:{round(mAP_folds_mean, 4)}"
    )
    print(f"Running time: {(time.time()-epoch_start)/60**1}")

    if CFG.neplog:
        run["CV"].log(CV_mAP)
        run["CV_mean"].log(mAP_folds_mean)

    # Save oofs/metrics
    save_path = f"{output_dir}/oofs/O_{CFG.model_name[:8]}_{CFG.multihead}_{CFG.target_size}_{CFG.epochs}_{CFG.SD}_{CFG.owner}_{CFG.exp}.csv"
    
    if not CFG.debug:
        oof_df.to_csv(save_path)
    
    if CFG.neplog:
        run[CFG.exp].upload(save_path)

    # mAP per video
    for i, v in enumerate(vids):

        # Filter df
        vid_df = oof_df[oof_df["video"] == v]

        # Score metrics for the fold
        tri0_idx = int(valid_folds.columns.get_loc("tri0"))
        pred0_idx = int(valid_folds.columns.get_loc("0"))

        # Metric
        classwise = average_precision_score(
            vid_df.iloc[:, tri0_idx : tri0_idx + 100].values,
            vid_df.iloc[:, pred0_idx : pred0_idx + 100].values,
            average=None,
        )
        mean = np.nanmean(classwise)

        if CFG.neplog:
            run[f"VIDS"].log(mean)

    