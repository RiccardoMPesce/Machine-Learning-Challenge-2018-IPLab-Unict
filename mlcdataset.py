"""
unzioni per operare sul dataset
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
    Implementa la classe MLCDataset, per utilizzarla su pytorch
    """

    def __init__(self, base_path, file_list, test=False, transform=None):
        """
        - base_path: percorso alla cartella contente le immagini
        - file_list: percorso al file contente i nomi delle immagini
        - transform: trasformazione da applicare
        """

        self.base_path = base_path
        self.image_list = [list(row) for row in pd.read_csv(file_list).values.tolist()]
        self.transform = transform

        # flag per indicare se sono etichette da test
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

# Media e dev. standard pre computate attraverso algoritmo commentato sotto
mean = torch.tensor([0.3881, 0.3659, 0.3551])
std = torch.tensor([0.2088, 0.2086, 0.2085])

"""
Calcolo media e dev. standard

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

# Composizione della trasformazione da applicare
normalization = transforms.Compose([transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])
