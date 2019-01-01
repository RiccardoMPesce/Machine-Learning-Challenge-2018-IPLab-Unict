"""
Purpose: make a list of all the training samples
"""

import pandas as pd

from os import path
from numpy.random import shuffle

def make_list(training_samples, validation_samples, test_samples,
              training_source_path, validation_source_path, test_source_path):
    training_list = [list(row) for row in pd.read_csv(training_source_path).values.tolist()]
    validation_list = [list(row) for row in pd.read_csv(validation_source_path).values.tolist()]
    test_list = [list(row) for row in pd.read_csv(test_source_path).values.tolist()]

    training_samples = len(training_list) if training_samples > len(training_list) else training_samples
    validation_samples = len(validation_list) if validation_samples > len(validation_list) else validation_samples
    test_samples = len(test_list) if test_samples > len(test_list) else test_samples

    with open("training_set.csv", "w") as training:
        for i in range(training_samples):
            shuffle(training_list)
            temp = training_list.pop(0)
            training.write(temp[0] + ", " + str(temp[-1]) + "\n")
    
    with open("validation_set.csv", "w") as validation:
        for i in range(validation_samples):
            shuffle(validation_list)
            temp = training_list.pop(0)
            validation.write(temp[0] + ", " + str(temp[-1]) + "\n")

    with open("test_set.csv", "w") as test:
        for i in range(test_samples):
            shuffle(test_list)
            temp = test_list.pop(0)
            test.write(temp[0] + "\n")