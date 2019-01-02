"""
    Goal: given an image taken inside a supermarket, identify the position (store dept.)
    where it was taken.
    Solving this as a classification problem, where given a picture, the appropriate
    department marked in the map is returned.
"""

import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import makelist
import sys

import mlcdataset as mlc
from matplotlib import pyplot as plt

# Module containing the ResNet model
from torchvision.models import resnet

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim import Adam
from torch.autograd import Variable

# Modules for testing accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Constants used throughout the code 
LR = 0.01
M = 0.99
N_EPOCHS = 50

IMG_PATH = "dataset/images"

TRAINING_SET_FILE = "training_set.csv"
VALIDATION_SET_FILE = "validation_set.csv"
TEST_SET_FILE = "test_set.csv"

PREDICTIONS_FILE = "predictions.csv"

N_TRAINING_SAMPLES = 1000
N_VALIDATION_SAMPLES = 250
N_TEST_SAMPLES = 100

BATCH_SIZE = 32
N_WORKERS = 2

PRINT_EVERY = 10 

makelist.make_list(N_TRAINING_SAMPLES, N_VALIDATION_SAMPLES, N_TEST_SAMPLES,
                   "dataset/training_list.csv", "dataset/validation_list.csv",
                   "dataset/testing_list_blind.csv")

kwargs = {"num_classes": 16}

resnet_model = resnet.resnet18(pretrained=False, **kwargs)
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = Adam(lr=LR, momentum=M, params=resnet_model.parameters())

# Instancing variables
training_set = mlc.MLCDataset(IMG_PATH, TRAINING_SET_FILE, transform=mlc.normalization)
training_set_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)

validation_set = mlc.MLCDataset(IMG_PATH, VALIDATION_SET_FILE, transform=mlc.normalization)
validation_set_loader = DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)

test_set = mlc.MLCDataset(IMG_PATH, TEST_SET_FILE, transform=mlc.normalization)
test_set_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)

def train_model(model=resnet_model, optimizer=optimizer, epochs=N_EPOCHS, momentum=M, 
                loader=training_set_loader, print_every=PRINT_EVERY):
    """
    Training procedure
    """
    losses = []

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for i, batch in enumerate(loader):
            x = Variable(batch["image"], requires_grad=True)
            y = Variable(batch["label"])

            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            epoch_loss += loss.data.mean()

            optimizer.step()

        epoch_loss /= len(loader.dataset)

        losses.append(epoch_loss)

        if epoch % print_every == 0:
            print "(Training) Epoch: %d/%d. Iteration: %d/%d. Loss: %0.2f." \
            % (epoch + 1, epochs, i, len(loader), epoch_loss)

    return model, losses


def validate_model(model=resnet_model, optimizer=optimizer, epochs=N_EPOCHS, momentum=M, 
                validation_loader=validation_set_loader):
    pass

def test_model(model=resnet_model, epochs=N_EPOCHS, test_loader=test_set_loader):
    pass   

resnet_model, resnet_model_log = train_model()

print training_set_loader.dataset[1]["image"].shape