import hydra
import warnings
from global_var import config_name
from train import train_fnt

warnings.filterwarnings("ignore")


@hydra.main(config_name=config_name)
def train(CFG):
    """
    Main training script using Hydra.

    Args:
        CFG (config object): Configuration object for training.

    Returns:
        None
    """
    train_fnt(CFG)


if __name__ == "__main__":
    train()
