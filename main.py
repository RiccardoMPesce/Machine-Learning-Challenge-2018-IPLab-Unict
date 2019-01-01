"""
    Goal: given an image taken inside a supermarket, identify the position (store dept.)
    where it was taken.
    Solving this as a classification problem, where given a picture, the appropriate
    department marked in the map is returned.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch 
import makelist
import sys

import mlcdataset as mlc
from matplotlib import pyplot as plt

# Module containing the ResNet model
from torchvision.models import resnet

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
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

resnet_model = resnet.resnet50(pretrained=False)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(lr=LR, momentum=M, params=resnet_model.parameters())

# Instancing variables
training_set = mlc.MLCDataset(IMG_PATH, TRAINING_SET_FILE, transform=mlc.normalization)
training_set_loader = DataLoader(dataset=training_set, batch_size=32, num_workers=2, shuffle=True)

validation_set = mlc.MLCDataset(IMG_PATH, VALIDATION_SET_FILE, transform=mlc.normalization)
validation_set_loader = DataLoader(dataset=validation_set, batch_size=32, num_workers=2, shuffle=True)

test_set = mlc.MLCDataset(IMG_PATH, TEST_SET_FILE, transform=mlc.normalization)
test_set_loader = DataLoader(dataset=test_set, batch_size=32, num_workers=2, shuffle=True)

def train_model(model=resnet_model, optimizer=optimizer, epochs=N_EPOCHS):
    pass

def validate_model(model=resnet_model, optimizer=optimizer, epochs=N_EPOCHS):
    pass

def test_model(model=resnet_model, epochs=):
    pass
