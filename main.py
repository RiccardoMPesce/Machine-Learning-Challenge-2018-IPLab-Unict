"""
    Goal: given an image taken inside a supermarket, identify the position (store dept.)
    where it was taken.
    Solving this as a classification problem, where given a picture, the appropriate
    department marked in the map is returned.
"""

import torch 
import makelist
import sys

# Module containing the ResNet model
from torchvision.models import resnet

# Module containing the dataset
import mlcdataset as mlc

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.autograd import Variable

# Modules for testing accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Command Line parameters are the arguments to our makelist function
if len(sys.argv) != 7:
    print "Invalid number of arguments"
    exit()
else:
    call = {}
    
    n_training_samples = int(sys.argv[1])
    n_validation_samples = int(sys.argv[2])
    n_test_samples = int(sys.argv[3])
    training_source_path = sys.argv[4]
    validation_source_path = sys.argv[5]
    test_source_path = sys.argv[6]

    call["training_samples"] = n_training_samples
    call["validation_samples"] = n_validation_samples
    call["test_samples"] = n_test_samples
    call["training_source_path"] = training_source_path
    call["validation_source_path"] = validation_source_path
    call["test_source_path"] = test_source_path

    makelist.make_list(**call)

# Instancing variables
training_set = mlc.MLCDataset("dataset/images", "training_set.csv", transform=mlc.normalization)
training_set_loader = DataLoader(dataset=training_set, batch_size=32, num_workers=2, shuffle=True)

validation_set = mlc.MLCDataset("dataset/images", "validation_set.csv", transform=mlc.normalization)
validation_set_loader = DataLoader(dataset=validation_set, batch_size=32, num_workers=2, shuffle=True)

test_set = mlc.MLCDataset("dataset/images", "test_set.csv", transform=mlc.normalization)
test_set_loader = DataLoader(dataset=test_set, batch_size=32, num_workers=2, shuffle=True)
