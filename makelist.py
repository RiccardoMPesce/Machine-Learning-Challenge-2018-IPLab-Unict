"""
Purpose: make a list of all the training samples
"""

import pandas as pd

from os import path
from numpy.random import shuffle

def make_list(training_samples, validation_samples, test_samples, labeled_source_path, unlabeled_source_path):
    labeled_list = [list(row) for row in pd.read_csv(labeled_source_path).values]
    unlabeled_list = [list(row) for row in pd.read_csv(unlabeled_source_path).values]

    training_samples = int(round(70.0 / 100.0 * len(labeled_list))) if training_samples > int(round(70.0 / 100.0 * len(labeled_list))) else training_samples
    validation_samples = int(round(30.0 / 100.0 * len(labeled_list))) if validation_samples > int(round(30.0 / 100.0 * len(labeled_list))) else validation_samples
    test_samples = len(unlabeled_list) if test_samples > len(unlabeled_list) else test_samples

    with open("training_set.csv", "w") as training:
        for i in range(training_samples):
            shuffle(labeled_list)
            temp = labeled_list.pop(i)
            training.write(temp[0] + ", " + str(temp[1]) + "\n")
    
    with open("validation_set.csv", "w") as validation:
        for i in range(validation_samples):
            shuffle(labeled_list)
            temp = labeled_list.pop(i)
            validation.write(temp[0] + ", " + str(temp[1]) + "\n")

    with open("test_set.csv", "w") as test:
        for i in range(test_samples):
            shuffle(unlabeled_list)
            temp = unlabeled_list.pop(i)
            test.write(temp[0] + "\n")