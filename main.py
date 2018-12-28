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
    def __init__(self, input_channels=3, out_classes=16):
        """
        Image classification cnn
        """
        super(Net, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 18, 5)
        )

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
