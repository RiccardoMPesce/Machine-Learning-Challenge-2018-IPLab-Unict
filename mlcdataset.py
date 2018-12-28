"""
Dataset related module
~
Funzioni per operare sul dataset
"""

import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from os import path

class MLCDataset(Dataset):
    """
    Implement the MLCDataset class, so as to use it as a standart torch dataset
    ~
    Implementa la classe MLCDataset, per utilizzarla su pytorch
    """

    def __init__(self, base_path, file_list, transform=None):
        """
        - base_path: path to the folder containing the sample images
        - file_list: path to the file containing sample labels
        - transform: desired transform to apply
        """

        self.base_path = base_path
        self.image_list = [list(row) for row in pd.read_csv(file_list).values]
        self.transform = transform

    def __getitem__(self, index):
        image_name, image_label = self.image_list[index]

        image_path = path.join(self.base_path, image_name)
        image = Image.open(image_path)
      
        if self.transform is not None:
            image = self.transform(image)

        image_label = int(image_label)

        return {"image": image, "label": image_label, "path": image_path}

    def __len__(self):
        return len(self.image_list)


# dataset without normalization
dataset = MLCDataset("dataset/images", "labeled.csv", transform=transforms.ToTensor())

mean = torch.zeros(3)
std = torch.zeros(3)

for sample in dataset:
    mean[0] += sample["image"][0, :, :].mean()
    mean[1] += sample["image"][1, :, :].mean()
    mean[2] += sample["image"][2, :, :].mean()

    std[0] += sample["image"][0, :, :].std()
    std[1] += sample["image"][1, :, :].std()
    std[2] += sample["image"][2, :, :].std()

mean /= len(dataset)
std /= len(dataset)

print mean, std

# Composing the transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std),
                                transforms.Lambda(lambda x: x.view(-1))])

# Now we can create our normalized dataset
dataset = MLCDataset("dataset/images", "validation_set.csv", transform=transform)
