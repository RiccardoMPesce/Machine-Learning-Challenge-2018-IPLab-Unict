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
N_EPOCHS = 50

IMG_PATH = "dataset/images"

TRAINING_SET_FILE = "training_set.csv"
VALIDATION_SET_FILE = "validation_set.csv"
TEST_SET_FILE = "test_set.csv"

PREDICTIONS_FILE = "predictions.csv"

N_TRAINING_SAMPLES = 50
N_VALIDATION_SAMPLES = 10
N_TEST_SAMPLES = 20

BATCH_SIZE = 1
N_WORKERS = 4

PRINT_EVERY = 1

makelist.make_list(N_TRAINING_SAMPLES, N_VALIDATION_SAMPLES, N_TEST_SAMPLES,
                   "dataset/training_list.csv", "dataset/validation_list.csv",
                   "dataset/testing_list_blind.csv")

kwargs = {"num_classes": 16}

resnet_model = resnet.resnet18(pretrained=False, **kwargs)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(lr=LR, momentum=M, params=resnet_model.parameters())

# Istanziamento dei vari set, secondo le dimensioni riportate come costanti a inizio file
training_set = mlc.MLCDataset(IMG_PATH, TRAINING_SET_FILE, transform=mlc.normalization)
training_set_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)

validation_set = mlc.MLCDataset(IMG_PATH, VALIDATION_SET_FILE, transform=mlc.normalization)
validation_set_loader = DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)

test_set = mlc.MLCDataset(IMG_PATH, TEST_SET_FILE, transform=mlc.normalization, test=True)
test_set_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)

def train_model(model=resnet_model, optimizer=optimizer, epochs=N_EPOCHS, momentum=M, 
                loader=training_set_loader, print_every=PRINT_EVERY):
    """
    Procedura di training.
    Vengono ritornati, oltre al modello, solo le losses e le accuracies, per fini
    "statistici" di comparazione con la validazione
    """
    losses = []
    accuracies = []

    if torch.cuda.is_available():
        model = model.cuda()

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for i, batch in enumerate(loader):
            x = Variable(batch["image"], requires_grad=True)
            y = Variable(batch["label"])

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            epoch_loss += loss.data.item()
            epoch_accuracy += accuracy_score(y.data, output.max(1)[1].data)

            optimizer.step()

        epoch_loss /= len(loader)
        epoch_accuracy /= len(loader)

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        if epoch % print_every == 0:
            print "(Training) Epoch: %d/%d. Iteration: %d/%d. Loss: %.2f. Accuracy: %.2f" \
            % (epoch + 1, epochs, i + 1, len(loader), epoch_loss, epoch_accuracy)

    return model, {"losses": losses, "accuracies": accuracies}


def validate_model(model=resnet_model, optimizer=optimizer, epochs=N_EPOCHS, momentum=M, 
                loader=validation_set_loader):
    """
    Procedura di validazione: 
    Vengono ritornati, oltre al modello e accuracies & losses, l'f1-score, la matrice di confusione e la media
    degli f1 (mF1). Inoltre, per ogni procedura di validazione, verrà salvato il relativo modello, in modo
    da scegliere alla fine, quello che ha riportato migliori performances.
    """
    losses = []
    accuracies = []

    preds = []

    model.eval()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for i, batch in enumerate(loader):
            x = Variable(batch["image"], requires_grad=False)
            y = Variable(batch["label"], requires_grad=False)

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            output = model(x)

            preds.append(output.max(1)[1].data)
            
            loss = criterion(output, y)

            epoch_accuracy += accuracy_score(y.data, output.max(1)[1].data)
            epoch_loss += loss.data.item()

        epoch_loss /= len(loader)
        epoch_accuracy /= len(loader)

        print "\r[TEST] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" % \
                (epoch + 1, epochs, i, len(loader), epoch_loss, epoch_accuracy)

    preds = torch.Tensor(preds)
    if torch.cuda.is_available():
        preds = preds.cuda()

    f1 = f1_score(y, preds, average=None)
    cm = confusion_matrix(y, preds)
    mf1s = f1.mean()

    print "Confusion Matrix: " + cm
    print "F1: " + f1 
    print "mF1: " + mf1s

    return model, {"losses": losses, "accuracies": accuracies, "f1_score": f1, 
                   "confusion_matrix": cm, "mf1s": mf1s}

def test_model(model=resnet_model, epochs=N_EPOCHS, test_loader=test_set_loader):
    """
    Procedura di testing:
    Questa procedura serve solamente a creare il file predictions.csv,
    e siccome le immagini da test non sono etichettate, non è possibile
    fare statistiche su di esse.
    """
    pass   

resnet_model, resnet_model_log = train_model()
validated_model, validation_log = validate_model(resnet_model)
