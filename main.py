"""
    Goal: given an image taken inside a supermarket, identify the position (store dept.)
    where it was taken.
    Solving this as a classification problem, where given a picture, the appropriate
    department marked in the map is returned.
"""

import torch 
import makelist
import sys

from matplotlib import pyplot as plt

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

model = resnet.resnet50(pretrained=False)

def train_model(model, lr=0.01, epochs=20, momentum=0.9, 
                train_loader=training_set_loader,
                validation_loader=validation_set_loader, 
                test_loader=test_set_loader):
    """
    Training procedure: it takes the model as input, and returns a tuple
    whose first element is the model itself, while the second element is 
    another tuple containing (losses, accuracies), so statistical data
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr, momentum=momentum)

    loaders = {"train": train_loader, "validation": validation_set_loader, "test": test_set_loader}
    losses = {"train": [], "validation": [], "test": []}
    accuracies = {"train": [], "validation": [], "test": []}

    for e in range(epochs):
        for mode in ["train", "validation" "test"]:
            if mode == "train":
                batch = training_set_loader
                model.train()
            else:
                batch = test_set_loader
                model.eval()
    
            epoch_loss = 0
            epoch_accuracy = 0
            samples = 0

            for i, batch in enumerate(loaders[mode]):
                # Let's turn tensors into variables
                x = Variable(batch[0], requires_grad=(mode == "train"))
                y = Variable(batch[1])

                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()

                output = model(x)
                cost = criterion(output, y)

                if mode == "train":
                    cost.backward()
                    optimizer.step()
                    # Preventing gradients to sum
                    optimizer.zero_grad()

                score = accuracy_score(y.data, output.max(1)[1].data)

                epoch_loss += cost.data[0] * x.shape[0]
                epoch_accuracy += score * x.shape[0]
                samples += x.shape[0]

                print "\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % (mode, e + 1, epochs, i, len(loaders[mode]), epoch_loss, epoch_accuracy),

    return model, (losses, accuracies)

model, logs = train_model(model, epochs=100)