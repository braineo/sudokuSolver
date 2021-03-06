import cv2
import numpy as np
import sys

imageRoute = '../test_picture/sudoku1.jpg'
originImage = cv2.imread(imageRoute)
resizeFactor = 500.0 / originImage.shape[0]
originImage = cv2.resize(
    originImage, (0, 0), fx=resizeFactor, fy=resizeFactor, interpolation=cv2.INTER_NEAREST)
gray = cv2.cvtColor(originImage, cv2.COLOR_BGR2GRAY)

originCopy = originImage.copy()
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType,
# blockSize, C[, dst])
thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)

kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
thresh = cv2.dilate(thresh, kernel)

contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# find rectangle with maximum area
maxIndex, maxArea = -1, 0
for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) > maxArea and \
            len(cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True) * 0.02, True)) == 4:
        maxArea = cv2.contourArea(cnt)
        maxIndex = i

poly = cv2.approxPolyDP(
    contours[maxIndex], cv2.arcLength(contours[maxIndex], True) * 0.02, True)
# cv2.polylines(originImage, [poly], True, (0, 0, 200), 4)

# -------------------------perspective transform ------------------------------- #
# transform polygon into points
poly = [i[0] for i in poly]

srtPoints = np.zeros((4, 2), dtype=np.float32)
# top-left smallest coordinate sum
# bottom-right largest coordinate sum
s = np.sum(poly, axis=1)
srtPoints[0] = poly[np.argmin(s)]
srtPoints[2] = poly[np.argmax(s)]

# top-right
d = np.diff(poly, axis=1)
srtPoints[1] = poly[np.argmin(d)]
srtPoints[3] = poly[np.argmax(d)]

extractedPoint = np.array([[0, 0],
                           [314, 0],
                           [314, 314],
                           [0, 314]], dtype=np.float32)

transMatrix = cv2.getPerspectiveTransform(srtPoints, extractedPoint)
warped = cv2.warpPerspective(originImage, transMatrix, (315, 315))

# -------------------------pick up digits------------------------------- #
gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
thresh = cv2.dilate(thresh, kernel)

digits = []
for i in xrange(0, 315, 35):
    for j in xrange(0, 315, 35):
        frag = thresh[i:i + 35, j:j + 35]
        frag = cv2.fastNlMeansDenoising(frag)
        frag = cv2.resize(frag, (28, 28))
        digits.append(frag)

trainData = np.load('../training_data/imageTrain.npy')
trainLabel = np.load('../training_data/labelTrain.npy')

trainData = trainData.reshape(60000, 784).astype(np.float32)
trainLabel = trainLabel.reshape(-1).astype(np.float32)

knn = cv2.KNearest()
knn.train(trainData, trainLabel)

for i in digits:
    i = cv2.bitwise_not(i, i)
    i = cv2.resize(i, (500,500))
    iCopy = i.copy()
    i = cv2.cvtColor(i, cv2.COLOR_GRAY2BGR)

    mask = np.zeros(i.shape, i.dtype)
    contours, hierarchy = cv2.findContours(iCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for j, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 40000:
            cv2.drawContours(mask, contours, j, (255, 255, 255), cv2.cv.CV_FILLED) # draw all contours
    cv2.bitwise_and(i, mask, i)
    cv2.imshow('image', i)
    cv2.waitKey(0)
test = i.reshape(1, 784).astype(np.float32)
image = i
retval, results, neighborResponses, dists = knn.find_nearest(test, k=5)
print retval, results, neighborResponses, dists

