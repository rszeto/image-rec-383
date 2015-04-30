# Useful general Python modules
from __future__ import division
import math
from os.path import isfile

# Useful modules for working with matrices and ML models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

import DT_test
import KNN_test
import LDA_test
import LR_test
import NB_test

np.random.seed(0)

# Load training and test data
train = np.load('train.npy')
test = np.load('test.npy')
np.random.shuffle(train)

# Get indexes for dividing training data into model training
# and validation sets
trainRows = range(0, 50000)
# Get feature vectors
trainData = train[:, 1:]
testData = test[:, 1:]
# Get labels
trainLabels = train[:, 0]
testLabels = test[:, 0]

# DT_test.run_test(trainData, trainLabels, testData, testLabels)

# # takes very long time (~12 mins)
# KNN_test.run_test(trainData, trainLabels, testData, testLabels)

# NB_test.run_test(trainData, trainLabels, testData, testLabels)

# LDA_test.run_test(trainData, trainLabels, testData, testLabels)

# LR_test.run_test(trainData, trainLabels, testData, testLabels)