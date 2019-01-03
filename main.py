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
N_EPOCHS = 10

IMG_PATH = "dataset/images"

TRAINING_SET_FILE = "training_set.csv"
VALIDATION_SET_FILE = "validation_set.csv"
TEST_SET_FILE = "test_set.csv"

PREDICTIONS_FILE = "predictions.csv"

N_TRAINING_SAMPLES = 60
N_VALIDATION_SAMPLES = 20
N_TEST_SAMPLES = 20

BATCH_SIZE = 1
N_WORKERS = 2

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

# resnet_model, resnet_model_log = train_model()
# validated_model, validation_log = validate_model(resnet_model)

model = resnet_model
epochs = N_EPOCHS
lr = LR

criterion = nn.CrossEntropyLoss()
#l'optimizer ci permetterà di effettuare la Stochastic Gradient Descent
#####################################################
#Specifichiamo un momentum pari a 0.9
optimizer = SGD(model.parameters(), lr, momentum=M)
#####################################################
training_losses = []
training_accuracies = []
test_losses = []
test_accuracies = []

for e in range(epochs):
    #ciclo di training
    model.train()
    train_loss = 0
    train_acc = 0
    for i, batch in enumerate(training_set_loader):
        #trasformiamo i tensori in variabili
        x = Variable(batch["image"])
        y = Variable(batch["label"])
        output = model(x)
        l = criterion(output,y)
        l.backward()

        acc = accuracy_score(y.data,output.max(1)[1].data)

        #accumuliamo i valori di training e loss
        #moltiplichiamo per x.shape[0], che restituisce la dimensione
        #del batch corrente.
        train_loss += l.data[0] * x.shape[0]
        train_acc += acc * x.shape[0]

        print "\r[TRAIN] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" % \
        (e + 1, epochs, i, len(training_set_loader), l.data[0], acc),

        optimizer.step() #sostituisce il codice di aggiornamento manuale dei parametri
        optimizer.zero_grad() #sostituisce il codice che si occupava di azzerare i gradienti

    train_loss /= len(training_set)
    train_acc /= len(training_set)

    training_losses.append(train_loss)
    training_accuracies.append(train_acc)

    print "\r[TRAIN] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" % \
    (e + 1, epochs, i, len(training_set_loader), train_loss, train_acc)
    #ciclo di test
    model.eval()
    test_acc = 0
    test_loss = 0
    for i, batch in enumerate(test_set_loader):
    #trasformiamo i tensori in variabili
        x = Variable(batch["image"], requires_grad=False)
        y = Variable(batch["label"], requires_grad=False)
        output = model(x)
        l = criterion(output,y)

        test_acc += accuracy_score(y.data,output.max(1)[1].data) * x.shape[0]
        test_loss += l.data[0] * x.shape[0]

        print "\r[TEST] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" % \
        (e + 1, epochs, i, len(training_set_loader), l.data[0], acc),

    #salviamo il modello
    torch.save(model.state_dict(),'model-%d.pth'%(e+1,))

    test_loss /= len(test_set)
    test_acc /= len(test_set)
    
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print "\r[TEST] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f" % \
    (e + 1, epochs, i, len(test_set_loader), test_loss, test_acc)
