import os
import hydra
import warnings
import pandas as pd

from preprocess import get_folds
from helper import inference_fn, get_inference_loader
from global_var import config_name
from utils import save_predictions
from models import get_pretrained_model

warnings.filterwarnings("ignore")


def inference(CFG):
    """
    Run inference and generate soft-labels or final predictions.

    Args:
        CFG (OmegaConf): Configuration object.

    Returns:
        None

    Side Effects:
        - Saves predictions to CSV file if CFG.inference is True, in the 'predictions' directory.
        - Saves soft-labels to CSV file if CFG.inference is False, in the 'softlabels' directory.
    """

    # Get folds and video ids
    folds, vids = get_folds(CFG)

    # Create soft labels folder
    softlabels_dir = os.path.join(CFG.output_dir, "softlabels")
    if not os.path.exists(softlabels_dir):
        os.mkdir(softlabels_dir)
        print("./softlabels directory created!")

    # Create predictions folder
    predictions_dir = os.path.join(CFG.output_dir, "predictions")
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)
        print("./predictions directory created!")

    # Initialize an empty dataframe for predictions
    pred_df = pd.DataFrame()

    if CFG.inference:
        print("\033[94mStarting inference\033[0m")
    else:
        print("\033[94mGenerating soft-labels\033[0m")

    # Process each fold
    for fold in range(CFG.n_fold):

        # Load pretrained weights
        model = get_pretrained_model(fold, CFG)

        # Get inference loader
        inference_folds, inference_loader = get_inference_loader(CFG, fold, folds)

        # Inference loop
        preds = inference_fn(CFG, inference_loader, model, CFG.device)

        # Load and save preds
        inference_folds[[str(c) for c in range(CFG.target_size)]] = preds

        # save path
        if CFG.inference:
            # Concatenate the folds predictions
            pred_df = pd.concat([pred_df, inference_folds])
        else:
            # Save soft-labels
            save_path = os.path.join(
                CFG.output_dir,
                f"softlabels/sl_f{fold}_{CFG.model_name[:8]}_{CFG.target_size}.csv",
            )
            inference_folds.to_csv(save_path)
            print(f"Soft labels saved at {save_path}")

    # Save final predictions of pretrained and custom models
    save_predictions(CFG, pred_df)


# Run the code
@hydra.main(config_name=config_name)
def generate(CFG):
    """
    Main function to run the inference.

    Args:
        CFG (OmegaConf): Configuration object.

    Returns:
        None
    """
    inference(CFG)


if __name__ == "__main__":
    generate()
