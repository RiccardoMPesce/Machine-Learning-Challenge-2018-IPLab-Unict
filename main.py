# coding=utf-8

"""
    Obiettivo: data un'immagine scatta all'interno di un reparto
    del supermercato, ritornare il numero identificante il reparto
    stesso
"""

import torch

import makelist
import sys
import numpy as np

import mlcdataset as mlc
from matplotlib import pyplot as plt

# Modulo contenente la resnet
from torchvision.models import resnet

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.autograd import Variable

# Modules for testing accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Costanti determinanti le dimensioni e gli iperparametri
LR = 0.01
M = 0.99
N_EPOCHS = 25

IMG_PATH = "dataset/images"

TRAINING_SET_FILE = "training_set.csv"
VALIDATION_SET_FILE = "validation_set.csv"
TEST_SET_FILE = "test_set.csv"

PREDICTIONS_FILE = "predictions.csv"

N_TRAINING_SAMPLES = -1
N_VALIDATION_SAMPLES = -1
N_TEST_SAMPLES = -1

BATCH_SIZE = 32
N_WORKERS = 4

PRINT_EVERY = 1

makelist.make_list(N_TRAINING_SAMPLES, N_VALIDATION_SAMPLES, N_TEST_SAMPLES,
                   "dataset/training_list.csv", "dataset/validation_list.csv",
                   "dataset/testing_list_blind.csv")

kwargs = {"num_classes": 16}

resnet18_model = resnet.resnet18(pretrained=False, **kwargs)
resnet34_model = resnet.resnet34(pretrained=False, **kwargs)

criterion = nn.CrossEntropyLoss()

optimizer_18 = SGD(lr=LR, momentum=M, params=resnet18_model.parameters())
optimizer_34 = SGD(lr=LR, momentum=M, params=resnet34_model.parameters())

# Istanziamento dei vari set, secondo le dimensioni riportate come costanti a inizio file
training_set = mlc.MLCDataset(IMG_PATH, TRAINING_SET_FILE, transform=mlc.normalization)
training_set_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)

validation_set = mlc.MLCDataset(IMG_PATH, VALIDATION_SET_FILE, transform=mlc.normalization)
validation_set_loader = DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)

test_set = mlc.MLCDataset(IMG_PATH, TEST_SET_FILE, transform=mlc.normalization, test=True)
test_set_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)

def train_model(model, optimizer, lr=LR, epochs=N_EPOCHS, momentum=M, 
                training_loader=training_set_loader,test_loader=validation_set_loader,
                criterion=criterion):
    loaders = {"training": training_loader, "test": test_loader}
    losses = {"training": [], "test": []}
    accuracies = {"training": [], "test": []}
    f1s = []
    cms = []

    if torch.cuda.is_available():
        model = model.cuda()

    for epoch in range(epochs):
        for mode in ["training", "test"]:
            if mode == "training":
                model.train()
            else:
                model.eval()

            epoch_loss = 0
            epoch_accuracy = 0
            samples = 0

            for i, batch in enumerate(loaders[mode]):
                x = Variable(batch["image"], requires_grad=(mode == "training"))
                y = Variable(batch["label"])

                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()

                output = model(x)
                loss = criterion(output, y)

                if mode == "training":
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                accuracy = accuracy_score(y.data, output.max(1)[1].data)

                epoch_loss += loss.data.item() * x.shape[0]
                epoch_accuracy += accuracy * x.shape[0]

                samples += x.shape[0]

                """ print "[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\n" % \
                    (mode, epoch + 1, epochs, i, len(loaders[mode]), epoch_loss / samples, epoch_accuracy / samples) """

            epoch_loss /= samples 
            epoch_accuracy /= samples

            losses[mode].append(epoch_loss)
            accuracies[mode].append(epoch_accuracy)

            print "[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\n" % \
                    (mode, epoch + 1, epochs, i, len(loaders[mode]), epoch_loss, epoch_accuracy)

    model_name = ""
    
    f1 = f1_score(y.data, output.max(1)[1].data, average=None)
    cm = confusion_matrix(y.data, output.max(1)[1].data)

    print "F1_score: " + str(f1)
    print "Confusion Matrix: " + str(cm)

    return model, {"losses": losses, "accuracies": accuracies, "f1": f1, "cm": cm}

resnet18_model, logs = train_model(model=resnet18_model, optimizer=optimizer_18)
