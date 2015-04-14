# Useful general Python modules
from __future__ import division
import math
import os

# Useful modules for working with matrices and ML models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_mldata

# Module for building the MNIST data sets
import datasetBuilder

############# Part of the script that does stuff #############

# Create loadable training and test set files if they do not exist
if not os.path.isfile('train.npy') or not os.path.isfile('test.npy'):
    print 'Did not find data sets. Creating them now.'
    datasetBuilder.createDatasets()
    print 'Finished creating data sets. Now analyzing data.'
else:
    print 'Data sets already exist. Now analyzing data.'

# Load training and test data
train = np.load('train.npy')
test = np.load('test.npy')
# Get an image and reshape it from 784x1 vector to 28x28 matrix
image = train[0, 1:].reshape((28, 28))
# Show the image
plt.gray()
plt.imshow(image)
plt.show()

# Get indexes for dividing training data into model training
# and validation sets
trainRows = range(0, 50000)
validationRows = range(50000, 60000)
# Get feature vectors
trainData = train[trainRows, 1:]
validationData = train[validationRows, 1:]
testData = test[:, 1:]
# Get labels
trainLabels = train[trainRows, 0]
validationLabels = train[validationRows, 0]
testLabels = test[:, 0]

# Initialize model
classifier = GaussianNB()
# Fit the model to the training data
classifier.fit(trainData, trainLabels)
# Evaluate the accuracy of the learned model on the validation set
score = classifier.score(validationData, validationLabels)
print score
