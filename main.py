from train import train_fnt
import hydra
import warnings

warnings.filterwarnings("ignore")


@hydra.main(config_name="config")
def train(CFG):
    train_fnt(CFG)


if __name__ == "__main__":
    train()
