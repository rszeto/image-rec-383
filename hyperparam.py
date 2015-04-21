# This module runs different classifiers over multiple sets of
# hyperparameters.

# Useful general Python modules
from __future__ import division
import math
from os.path import isfile
from time import time

# Useful modules for working with matrices and ML models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.grid_search import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Module for building the MNIST data sets
import datasetBuilder

############# Part of the script that does stuff #############

# Create loadable training and test set files if they do not exist
if not isfile('train.npy') or not isfile('test.npy'):
    print 'Did not find data sets. Creating them now.'
    datasetBuilder.createDatasets()
    print 'Finished creating data sets. Now analyzing data.'
else:
    print 'Data sets already exist. Now analyzing data.'
        
# Set random seed so we can replicate results
np.random.seed(0)

# Load training and test data
train = np.load('train.npy')
test = np.load('test.npy')

# Get indexes for dividing training data into model training
# and validation sets
trainRows = range(0, 50000)
validationRows = range(50000, 60000)
# Shuffle the training data
np.random.shuffle(train)
# Get feature vectors
trainData = train[trainRows, 1:]
validationData = train[validationRows, 1:]
testData = test[:, 1:]
fullTrainData = train[:, 1:]
# Get labels
trainLabels = train[trainRows, 0]
validationLabels = train[validationRows, 0]
testLabels = test[:, 0]
fullTrainLabels = train[:, 0]

# Search through a given set of hyperparameters for the given class, and
# return the parameters for the model with the best validation accuracy
def hyperparamSearch(modelClass, paramGrid, trainData, trainLabels, validData, validLabels):
    # Convert paramGrid to list so we can index over it
    paramGrid = list(paramGrid)
    # Store the score of each iteration
    regScores = np.zeros(len(paramGrid))
    for i in range(len(paramGrid)):
        # Get the hyperparameters for this iteration
        paramSet = paramGrid[i]
        print 'Parameters:'
        print paramSet
        # Initialize model with hyperparameters
        classifier = modelClass(**paramSet)
        startTime = time()
        # Train model
        classifier.fit(trainData, trainLabels)
        # Score model on validation data
        regScores[i] = classifier.score(validationData, validationLabels)
        endTime = time()
        print 'Score: %f' % regScores[i]
        print 'Validation duration: %d' % (endTime-startTime)
    # Find the iteration with the best score
    bestI = regScores.argmax()
    bestScore = regScores[bestI]
    # Get best set of hyperparameters
    bestParamSet = paramGrid[bestI]
    print 'Best parameter set with score of %f:' % bestScore
    print bestParamSet
    return {'paramSet': bestParamSet, 'score': bestScore}

# Hyperparameter selection for decision trees
print 'Doing hyperparameter selection over decision trees'
criterion_range = ['gini', 'entropy']
max_features_range = [None, 'sqrt']
gridDict = {'criterion': criterion_range, 'max_features': max_features_range}
pGrid = ParameterGrid(gridDict)
bestTreeInfo = hyperparamSearch(DecisionTreeClassifier, pGrid, trainData, trainLabels, validationData, validationLabels)

# Get accuracy on test data for best tree
model = DecisionTreeClassifier(**bestTreeInfo['paramSet'])
model.fit(fullTrainData, fullTrainLabels)
score = model.score(testData, testLabels)
print 'Test score for best tree: %f' % score

# Hyperparameter selection for KNN
print 'Doing hyperparameter selection over KNN'
K_range = range(1, 21, 2)
weights_range = ['uniform', 'distance']
metric_range = ['euclidean']
gridDict = {'n_neighbors': K_range, 'weights': weights_range, 'metric': metric_range}
pGrid = ParameterGrid(gridDict)
bestKNNInfo = hyperparamSearch(KNeighborsClassifier, pGrid, trainData, trainLabels, validationData, validationLabels)
print bestKNNInfo

# Get accuracy on test data for best tree
model = KNeighborsClassifier(**bestKNNInfo['paramSet'])
model.fit(fullTrainData, fullTrainLabels)
score = model.score(testData, testLabels)
print 'Test score for best KNN: %f' % score