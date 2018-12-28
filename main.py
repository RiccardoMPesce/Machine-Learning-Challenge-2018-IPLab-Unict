"""
    Goal: given an image taken inside a supermarket, identify the position (store dept.)
    where it was taken.
    Solving this as a classification problem, where given a picture, the appropriate
    department marked in the map is returned.
"""

import torch 
import makelist
import sys

from torch import nn

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

cnn = Net()

