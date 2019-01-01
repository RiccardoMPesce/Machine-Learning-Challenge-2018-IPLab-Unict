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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLCDataset(Dataset):
    """
    Implement the MLCDataset class, so as to use it as a standart torch dataset
    ~
    Implementa la classe MLCDataset, per utilizzarla su pytorch
    """

    def __init__(self, base_path, file_list, test=False, transform=None):
        """
        - base_path: path to the folder containing the sample images
        - file_list: path to the file containing sample labels
        - transform: desired transform to apply
        """

        self.base_path = base_path
        self.image_list = [list(row) for row in pd.read_csv(file_list).values.tolist()]
        self.transform = transform

        # Is in test mode?
        self.test = test

    def __getitem__(self, index):
        sample = self.image_list[index]
        image_name = sample[0]

        if not self.test:
            image_label = sample[1]

        image_path = path.join(self.base_path, image_name)
        image = Image.open(image_path)
      
        if self.transform is not None:
            image = self.transform(image)

        if not self.test:
            image_label = int(image_label)
            return {"image": image, "label": image_label}
        else:
            return {"image": image}

    def __len__(self):
        return len(self.image_list)

# Precomputed mean and std so as to save time.
# Code to calculate them is commented beneath
mean = torch.tensor([0.3881, 0.3659, 0.3551])
std = torch.tensor([0.2088, 0.2086, 0.2085])

"""
How to calculate mean and std

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
"""

# Composing the transform
normalization = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std),])

training_set = MLCDataset("dataset/images", "training_set.csv", transform=normalization)
training_set_loader = DataLoader(dataset=training_set, batch_size=32, num_workers=2, shuffle=True)
