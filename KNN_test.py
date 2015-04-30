import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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
  # print trainData
  # print trainLabels
  start_time = time()
  classifier = KNeighborsClassifier(n_neighbors=3, metric="euclidean", weights="distance")
  classifier.fit(trainData, trainLabels)
  score = classifier.score(testData, testLabels)
  duration = time() - start_time
  print "training set size: " + str(len(trainData))
  print "score: " + str(score)
  print "time: " + str(duration) + "\n"

# score: 97.05 %
# time: 12.027 mins

for i in range(10000, 60001, 10000):
  trainData = train[0:i, 1:]
  trainLabels = train[0:i, 0]
  run_test(trainData, trainLabels, testData, testLabels)