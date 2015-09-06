__author__ = 'braineo'

import cv2
import numpy as np

originImage = cv2.imread('sudoku1.jpg')
resizeFactor = 500.0/originImage.shape[0]
originImage = cv2.resize(originImage, (0, 0), fx=resizeFactor, fy=resizeFactor, interpolation=cv2.INTER_NEAREST)
gray = cv2.cvtColor(originImage, cv2.COLOR_BGR2GRAY)

originCopy = originImage.copy()
blur = cv2.GaussianBlur(gray, (7,7), 0)

# cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)

kernel = np.array([[0,1,0], [1,1,1], [0,1,0]],np.uint8)
thresh = cv2.dilate(thresh, kernel)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# find rectangle with maximum area
maxIndex, maxArea = -1, 0
for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) > maxArea and \
                  len(cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True) * 0.02, True)) == 4:
        maxArea = cv2.contourArea(cnt)
        maxIndex = i

cv2.drawContours(originImage, contours, maxIndex, (0, 0, 200), 4)
cv2.imshow("poly", originImage)
cv2.waitKey(0)
