import cv2
import numpy as np


    
img = cv2.imread('imgs/28.45_0.png', cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(img,img, mask= mask)

cv2.imshow('frame',img)
cv2.imshow('mask',mask)
cv2.imshow('res',res)

cv2.imwrite("imgs/" + "res.png", res)
    
while(1):
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
