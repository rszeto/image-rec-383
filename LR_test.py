import numpy as np
from sklearn.linear_model import LogisticRegression
from time import time

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

def run_test(trainData, trainLabels, testData, testLabels):
  start_time = time()
  classifier = LogisticRegression(penalty='l2')
  classifier.fit(trainData, trainLabels)
  score = classifier.score(testData, testLabels)
  duration = time() - start_time
  print score
  print duration