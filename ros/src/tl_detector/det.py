import cv2
import numpy as np


    
img = cv2.imread('imgs/033.09_2.png', cv2.IMREAD_COLOR)
output = img.copy()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([0,150,100])
upper_red = np.array([20,255,255])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(img,img, mask= mask)

# detect circles in the image
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
cv2.imshow("mask", res)
#circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 30, 20, minRadius= 5, maxRadius=40)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 30, minRadius= 10, maxRadius=40) 

# # ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
	# show the output image
	cv2.imshow("output", np.hstack([img, output]))
cv2.waitKey(0)

# cv2.imshow('frame',img)
# cv2.imshow('mask',mask)
# cv2.imshow('res',res)

# cv2.imwrite("imgs/" + "res.png", res)
    
# while(1):
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break

# cv2.destroyAllWindows()
