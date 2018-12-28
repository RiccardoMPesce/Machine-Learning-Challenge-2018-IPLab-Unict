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
dataset = MLCDataset("dataset/images", "labeled.csv", transform=None)

mean = np.zeros(3)
var = np.zeros(3)

for sample in dataset:
    img_pix = np.asarray(sample["image"].convert("RGB"))
    
    red_channel = img_pix[:, :, 0]
    green_channel = img_pix[:, :, 1]
    blue_channel = img_pix[:, :, 2]

    mean[0] += red_channel.mean()
    mean[1] += green_channel.mean()
    mean[2] += blue_channel.mean()

    var[0] += red_channel.var()
    var[1] += green_channel.var()
    var[2] += blue_channel.var()

mean /= dataset.__len__()
var /= dataset.__len__()

# Composing the transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, var),
                                transforms.Lambda(lambda x: x.view(-1))])

# Now we can create our normalized dataset
unnorm = MLCDataset("dataset/images", "validation_set.csv", transform=transforms.ToTensor())
norm = MLCDataset("dataset/images", "validation_set.csv", transform=transform)

# Optional test
print unnorm[0]["image"]
print norm[0]["image"]
