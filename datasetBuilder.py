'''
This script converts mnist_all.mat into two NumPy files that contain
the training and test data in a convenient, directly loadable format.
The format for the produced NumPy arrays is as follows.

Each row corresponds to one data case. For each row, the first entry is
the classification label (0-9). The rest of the row contains the values
of the feature vector for the data point. The values in the feature
vector correspond to the brightnesses of each pixel (0-255).

The produced training set is a 60000x785 matrix, and the produced test
set is a 10000x785 matrix.

This can be run either by calling "python datasetBuilder.py" from the
command line or by importing this module and calling the createDatasets
function.
'''

import numpy as np
from scipy.io import loadmat

def createDatasets():
    # Load .mat file with training/test split
    mat = loadmat('mnist_all.mat')

    # Build training set
    fullTrainSetArrays = ()
    for i in range(10):
        trainSet = mat['train' + str(i)]
        numImages = trainSet.shape[0]
        labelVec = np.zeros((numImages, 1))
        labelVec.fill(i)
        newSet = np.concatenate((labelVec, trainSet), axis=1)
        fullTrainSetArrays = fullTrainSetArrays + (newSet,)
    fullTrainSet = np.concatenate(fullTrainSetArrays)
    np.save('train', fullTrainSet)

    # Build test set
    fullTestSetArrays = ()
    for i in range(10):
        testSet = mat['test' + str(i)]
        numImages = testSet.shape[0]
        labelVec = np.zeros((numImages, 1))
        labelVec.fill(i)
        newSet = np.concatenate((labelVec, testSet), axis=1)
        fullTestSetArrays = fullTestSetArrays + (newSet,)
    fullTestSet = np.concatenate(fullTestSetArrays)
    np.save('test', fullTestSet)

if __name__ == '__main__':
    createDatasets()