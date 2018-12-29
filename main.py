"""
    Goal: given an image taken inside a supermarket, identify the position (store dept.)
    where it was taken.
    Solving this as a classification problem, where given a picture, the appropriate
    department marked in the map is returned.
"""

import torch 
import makelist
import sys

# Module containing the dataset
import mlcdataset as mlc

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        """
        Image classification cnn
        Input: 
            - 3 channels (RGB)
        Output:
            - Number in (0, 15) 
        """
        super(Net, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 18, 5), # input: 3 x 256 x 144 -> output: 18 x 254 x 140
            nn.MaxPool2d(2), # input: 18 x 254 x 140 -> output: 18 x 127 x 69
            nn.ReLU(),
            nn.Conv2d(18, 28, 5), # input: 18 x 127 x 69 -> output: 28 x 123 x 65
            nn.MaxPool2d(3), # input: 28 x 123 x 65 -> output: 28 x 41 x 22
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(25256, 12650),
            nn.ReLU(),
            nn.Linear(12650, 6330),
            nn.ReLU(),
            nn.Linear(6330, 3160),
            nn.ReLU(),
            nn.Linear(3160, 15)
        )

    def forward(self, input):
        input = self.feature_extractor(input)
        input = self.classifier(input)

# Command Line parameters are the arguments to our makelist function
if len(sys.argv) != 6:
    print "Invalid number of arguments"
    exit()
else:
    call = {}
    call["training_samples"] = int(sys.argv[1])
    call["validation_samples"] = int(sys.argv[2])
    call["test_samples"] = int(sys.argv[3])
    call["labeled_source_path"] = sys.argv[4]
    call["unlabeled_source_path"] = sys.argv[5]

    makelist.make_list(**call)

net = Net()

# Instancing variables
training_set = mlc.MLCDataset("dataset/images", "training_set.csv", transform=mlc.normalization)
training_set_loader = DataLoader(dataset=training_set, batch_size=32, num_workers=2, shuffle=True)
validation_set = mlc.MLCDataset("dataset/images", "validation_set.csv", transform=mlc.normalization)
validation_set_loader = DataLoader(dataset=validation_set, batch_size=32, num_workers=2, shuffle=True)
test_set = mlc.MLCDataset("dataset/images", "test_set.csv", transform=mlc.normalization)
test_set_loader = DataLoader(dataset=test_set, batch_size=32, num_workers=2, shuffle=True)

def train_model(model, lr=0.01, epochs=20, momentum=0.9,
                train_loader=training_set_loader,
                test_loader=test_set_loader):
    """
    Training procedure: it takes the model as input, and returns a tuple
    whose first element is the model itself, while the second element is 
    another tuple containing (losses, accuracies), so statistical data
    """

    for mode in ("train", "test"):
        if mode == "train":
            model.train()
        else:
            model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr, momentum=momentum)

    
