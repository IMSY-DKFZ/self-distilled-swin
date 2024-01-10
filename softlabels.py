import torch
import os
from torch.utils.data import DataLoader
import hydra
import warnings
warnings.filterwarnings("ignore")

from models import TripletModel
from preprocess import get_folds
from augmentation import get_transforms
from dataset import TrainDataset
from helper import inference_fn


def inference(CFG):

    # Dataframe
    folds, vids = get_folds(CFG.n_fold, CFG)

    # Create soft labels folder
    if not os.path.exists(os.path.join(CFG.parent_path, "softlabels")):
        os.mkdir(os.path.join(CFG.parent_path, "softlabels"))

    # TRAINING FOLDS
    for fold in range(CFG.n_fold):
        print(f"Generating soft labels: fold {fold}")

        # Load model
        model = TripletModel(CFG, model_name=CFG.model_name, pretrained=False).to(
            CFG.device
        )

        # Load the weights
        weights = os.path.join(
            CFG.output_dir,
            f"output/checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.pth",
        )
        model.load_state_dict(torch.load(weights)["model"])
        print("Weights loaded successfully!")

        # Get train and valid indexes
        trn_idx = folds[folds["fold"] != fold].index

        # Get train dataframe
        train_folds = folds.loc[trn_idx].reset_index(drop=True)

        # Pytorch dataset
        train_dataset = TrainDataset(
            train_folds,
            transform=get_transforms(CFG=CFG, data="valid"),
            inference=True,
            CFG=CFG,
        )

        # Pytorch dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.nworkers,
            pin_memory=False,
            drop_last=False,
        )

        # VALIDATION LOOP
        preds = inference_fn(train_loader, model, CFG.device)

        # Load and save preds
        train_folds[[str(c) for c in range(CFG.target_size)]] = preds

        # Save soft-labels
        train_folds.to_csv(
            os.path.join(
                CFG.parent_path,
                f"softlabels/sl_f{fold}_{CFG.model_name[:8]}_{CFG.target_size}.csv",
            )
        )




# Run the code
@hydra.main(config_name="config")
def generate(CFG):
    inference(CFG)


if __name__ == "__main__":
    generate()
