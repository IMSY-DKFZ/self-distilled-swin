import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tri_index import index_by_occurrence

# Fix number of threads used by opencv
cv2.setNumThreads(1)


class TrainDataset(Dataset):
    """
    Custom PyTorch dataset for training.

    Parameters:
    - df (pd.DataFrame): DataFrame containing dataset information.
    - CFG (object): Configuration object containing hyperparameters.
    - transform (callable, optional): Optional data transformations. Default is None.
    - inference (bool, optional): Flag indicating inference mode. Default is False.

    Attributes:
    - df (pd.DataFrame): DataFrame containing dataset information.
    - CFG (object): Configuration object containing hyperparameters.
    - file_names (numpy.ndarray): Array containing file names.
    - transform (callable, optional): Optional data transformations.
    - inference (bool, optional): Flag indicating inference mode.
    - labels (torch.FloatTensor): Tensor containing target labels.

    Methods:
    - __len__(): Returns the length of the dataset.
    - __getitem__(index): Retrieves an item from the dataset.

    Example:
    ```python
    dataset = TrainDataset(df, CFG, transform=get_transforms(), inference=False)
    ```

    """

    def __init__(self, df, CFG, transform=None, inference=False):
        self.df = df
        self.CFG = CFG
        self.file_names = df["image_path"].values
        self.transform = transform
        self.inference = inference
        index_no = int(df.columns.get_loc(CFG.col0))
        self.labels = torch.FloatTensor(
            self.df.iloc[:, index_no : index_no + CFG.target_size].values.astype(
                np.float16
            )
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Parameters:
        - index (int): Index of the item to retrieve.

        Returns:
        - torch.Tensor or tuple: Image and target label if not in inference mode,
          otherwise only the image.

        """
        # Localize the image and targets
        file_name = self.file_names[index]
        target = self.labels[index]

        # Read the image
        file_path = os.path.join(self.CFG.parent_path, self.CFG.train_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.inference:
            return image
        else:
            return image, target
