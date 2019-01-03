# coding=utf-8

"""
Scopo: creare le liste da utilizzare durante l'allenamento, la validazione
e il test del nostro modello
"""

import pandas as pd

from os import path
from numpy.random import shuffle

def make_list(training_samples, validation_samples, test_samples,
              training_source_path, validation_source_path, test_source_path):
    training_list = [list(row) for row in pd.read_csv(training_source_path).values.tolist()]
    validation_list = [list(row) for row in pd.read_csv(validation_source_path).values.tolist()]
    test_list = [list(row) for row in pd.read_csv(test_source_path).values.tolist()]

    with open("training_set.csv", "w") as training:
        if training_samples <= 0 or training_samples > len(training_list):
            shuffle(training_list)
            for temp in training_samples:
                training.write(temp[0] + ", " + str(temp[-1]) + "\n")
        else:
            for i in range(training_samples):
                shuffle(training_list)
                temp = training_list.pop(0)
                training.write(temp[0] + ", " + str(temp[-1]) + "\n")
    
    with open("validation_set.csv", "w") as validation:
        if validation_samples <= 0 or validation_samples > len(validation_list):
            shuffle(validation_list)
            for temp in validation_list:
                validation.write(temp[0] + ", " + str(temp[-1]) + "\n")
        else:
            for i in range(validation_samples):
                shuffle(validation_list)
                temp = validation_list.pop(0)
                validation.write(temp[0] + ", " + str(temp[-1]) + "\n")

    with open("test_set.csv", "w") as test:
        if test_samples <= 0 or test_samples > len(test_list):
            shuffle(test_list)
            for temp in test_list:
                test.write(temp[0] + ", " + str(temp[-1]) + "\n")
        else:
            for i in range(test_samples):
                shuffle(test_list)
                temp = test_list.pop(0)
                test.write(temp[0] + "\n")