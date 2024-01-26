import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


import torch
import torch.nn as nn
import timm


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

        # Load the backbone
        self.model = timm.create_model(model_name, pretrained=pretrained)

        if CFG.local_weight:
            self.model.load_state_dict(
            torch.load(f"{CFG.weight_dir}/swin_base_patch4_window7_224_22kto1k.pth")[
                "model"
            ]
        )

        # Get the number features in final embedding
        n_features = self.model.head.in_features

        # Update the classification layer with our custom target size
        self.model.head = nn.Linear(n_features, CFG.target_size)

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
