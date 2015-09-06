import sys
import cv2
import numpy as np

im = cv2.imread('cbhsudoku.jpg')
im3 = im.copy()

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5, 5), 0)
thresh = cv2.adaptiveThreshold(blur,255, 1, 1, 11, 2)

contours, hierachy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0,100))
response = []
keys = set(range(ord('0'),ord('9')))

for cnt in contours:
    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)

        if h > 28:
            cv2.rectangle(im,(x,y), (x+w, y+h), (0,0,255),2)
            roi = thresh[y:y+h, x:x+w]
            roiSmall = cv2.resize(roi, (10, 10))
            cv2.imshow('norm', im)
            key = cv2.waitKey(0)

            if key == 27:
                sys.exit()
            elif key in keys:
                response.append(key - chr('0'))
                sample = roiSmall.reshape((1, 100))
                samples = np.append(samples, sample, 0)

response = np.array(response, np.float32)
response = response.reshape((response.size, 1))

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)