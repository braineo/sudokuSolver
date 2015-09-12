'''
train K-Nearest Neighbor classifier for digit recognition
'''
import cv2
import numpy as np

trainData = np.load('../training_data/imageTrain.npy')
trainLabel = np.load('../training_data/labelTrain.npy')

trainData = trainData.reshape(60000, 784).astype(np.float32)
trainLabel = trainLabel.reshape(-1).astype(np.float32)

knn = cv2.KNearest()
knn.train(trainData, trainLabel)

testData = np.load('../training_data/imageTest.npy')
testData = testData.reshape(10000, 784).astype(np.float32)
test = testData[0].reshape(1,784)

retval, results, neighborResponses, dists = knn.find_nearest(test, k = 5)
print retval, results, neighborResponses, dists

testLabel = np.load('../training_data/labelTest.npy')
print testLabel[0]