import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import os

from utils import update_model_config


class TripletModel(nn.Module):
    """
    Custom PyTorch model for the triplet classification task.

    Parameters:
    - CFG (object): Configuration object containing hyperparameters.
    - model_name (str): Name of the pre-trained model architecture.
    - pretrained (bool, optional): Flag indicating whether to use pre-trained weights. Default is True.

    Attributes:
    - model (nn.Module): Pre-trained model backbone.
    - n_features (int): Number of features in the final embedding.

    Methods:
    - forward(x): Forward pass through the model.

    """

    def __init__(self, CFG, model_name, pretrained=True):
        super().__init__()

        """
        Models class to return swin transformer models
        """
        self.CFG = CFG
        # Load the backbone
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=CFG.target_size,
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.

        """
        x = self.model(x)


        return x



def get_pretrained_model(fold, CFG):
    """
    Load a pretrained model or custom weights based on the specified configuration.

    Args:
        fold (int): The fold number.
        CFG (config object): Configuration object containing model settings.

    Returns:
        torch.nn.Module: The loaded model.
    """

    # Available pretrained models
    pretrained_models = [
        "SwinT",
        "SwinT+MultiT",
        "SwinT+SelfDv2",
        "SwinT+MultiT+SelfD",
        "+phase",
        "SwinLarge",
    ]

    # Update target size and model name if pretrained_model is True
    if CFG.pretrained_model:
        update_model_config(CFG)

    # Initialize the model
    model = TripletModel(CFG, model_name=CFG.model_name, pretrained=False).to(
        CFG.device
    )

    # Download pretrained weights or load custom weights
    if CFG.pretrained_model:
        if CFG.exp not in pretrained_models:
            raise Exception(
                f"Requested model: exp={CFG.exp} is not available, please select one of the available models:\n{pretrained_models}"
            )

        # Update the fold and exp tag to match the experiment
        checkpoint_url=f"https://self-distillation-weights.s3.dkfz.de/fold{fold}_{CFG.exp}.pth"
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)
        model.load_state_dict(checkpoint["model"])

        print(f"fold {fold}: Pretrained Weights downloaded and loaded successfully")

    else:

        # Load your custom weights
        weights_path = os.path.join(
            CFG.output_dir,
            f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.pth",
        )
        model.load_state_dict(torch.load(weights_path)["model"])
        print(f"fold {fold}: Weights loaded successfully")

    return model
