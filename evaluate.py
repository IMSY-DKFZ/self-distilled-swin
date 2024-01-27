import pandas as pd
import os
import hydra
import warnings
from utils import cholect45_ivtmetrics_mAP

warnings.filterwarnings('ignore')


def evaluate(CFG):
    """
    Evaluate predictions using the CholecT45 metric for experiments.

    This function reads prediction files from a specified folder, computes the CholecT45 metric for each experiment,
    and optionally computes the ensemble metric if specified in the configuration.

    Args:
        CFG (OmegaConf): Configuration object.

    Returns:
        None
    """
    # Set target size to 100 to evaluate on the triplets only
    CFG.target_size = 100

    # Determine the folder of saved predictions (inference or out-of-folds)
    folder = "predictions" if CFG.inference else "oofs"

    # Get the available experiments in the specified folder
    prediction_dfs = os.listdir(os.path.join(CFG.output_dir, folder))

    # Loop over the experiments
    for pred_df in prediction_dfs:
        # Load the dataframe
        df = pd.read_csv(os.path.join(CFG.output_dir, folder, pred_df))

        # Parse the experiment tag
        experiment = pred_df.split(".")[0].split('_')[-1]

        # Get the mAP score
        score = cholect45_ivtmetrics_mAP(df, CFG)
        print(f"{experiment}: {round(score * 100, 2)}")

    # Compute the ensemble of multiple experiments available in CFG.ensemble_models
    if CFG.ensemble:
        try:
            preds = None
            for model in CFG.ensemble_models:
                # Load the model's predictions
                df = pd.read_csv(os.path.join(CFG.output_dir, folder, model))

                # Get the indexes of the 1st prediction columns
                pred0_idx = df.columns.get_loc("0")

                # Accumulate the predictions
                preds = preds + df.iloc[:, pred0_idx:pred0_idx + 100].values if preds is not None else df.iloc[:, pred0_idx:pred0_idx + 100].values

            df.iloc[:, pred0_idx:pred0_idx + 100] = preds

            # Compute the ensemble mAP metric
            score = cholect45_ivtmetrics_mAP(df, CFG)

            # Get experiment tags for ensemble models
            ensemble_experiments = [model.split(".")[0].split('_')[-1] for model in CFG.ensemble_models]
            print(f"Ensemble of {ensemble_experiments}: {round(score * 100, 2)}")
        except Exception as e:
            print("Ensemble didn't work: Please check the spelling or the path of your prediction csv files.")
            print(e)



# Run the code
@hydra.main(config_name="config_amin")
def run(CFG):
    """
    Main function to run the evaluation.

    Args:
        CFG (OmegaConf): Configuration object.

    Returns:
        None
    """
    evaluate(CFG)


if __name__ == "__main__":
    run()
