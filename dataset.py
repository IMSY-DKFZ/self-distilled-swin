import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tri_index import index_by_occurrence
from torchvision import transforms


class TrainDataset(Dataset):
    def __init__(self, df, CFG, transform=None):
        self.df = df
        self.CFG = CFG
        self.file_names = df["nid"].values
        self.transform = transform
        index_no = int(df.columns.get_loc(CFG.col0))
        self.labels = torch.FloatTensor(
            self.df.iloc[:, index_no : index_no + CFG.target_size].values.astype(
                np.float16
            )
        )


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Localize the image and targets
        file_name = self.file_names[index]
        target = self.labels[index]


        # Read the image
        file_path = (
                f"{os.path.join(self.CFG.parent_path,self.CFG.TRAIN_PATH)}/{file_name}"
            )
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, target