from train import train_fnt
import hydra
import warnings
from global_var import config_name


warnings.filterwarnings("ignore")


@hydra.main(config_name=config_name)
def train(CFG):
    train_fnt(CFG)


if __name__ == "__main__":
    train()
