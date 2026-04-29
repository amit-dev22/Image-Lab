import cv2
import numpy as np

# Read input image
img = cv2.imread('C:/Users/kirti/OneDrive/Desktop/image processing lab/im.jpeg')

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range (HSV values)
lower = np.array([30, 50, 50])
upper = np.array([90, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower, upper)

# Apply mask
result = cv2.bitwise_and(img, img, mask=mask)

# Display images
cv2.imshow('Original', img)
cv2.imshow('Segmented', result)

cv2.waitKey(0)
cv2.destroyAllWindows()