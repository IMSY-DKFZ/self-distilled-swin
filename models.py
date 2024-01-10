import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class TripletModel(nn.Module):
    def __init__(self, CFG, model_name, pretrained=True):
        super().__init__()

        """
        Models class to return swin transformer models
        """

        # Load the backbone
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # Get the number features in final embedding
        n_features = self.model.head.in_features

        # Update the classification layer with our custom target size
        self.model.head = nn.Linear(n_features, CFG.target_size)

    # Forward pass
    def forward(self, x):
        x = self.model(x)
        return x
