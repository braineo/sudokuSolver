'''
train K-Nearest Neighbor classifier for digit recognition
'''
import cv2
import numpy as np

trainData = np.load('../training_data/imageTrain.npy')
trainLabel = np.load('../training_data/labelTrain.npy')
trainData = np.array(trainData, np.float32)
trainLabel = np.array(trainLabel, np.float32)
knn = cv2.KNearest()
knn.train(trainData, trainLabel)