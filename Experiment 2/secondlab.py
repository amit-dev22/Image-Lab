import cv2
import numpy as np

img = cv2.imread(r"C:\Users\kirti\OneDrive\Desktop\image processing lab\im.jpeg",0)

if img is None:
    print("Error loading image")
    exit()

sampled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

sampled_display = cv2.resize(sampled, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

levels = 8
step = 256 // levels
quantized = (img // step) * step


cv2.imshow("Original Image", img)
cv2.imshow("Sampled Image (Pixelated)", sampled_display)
cv2.imshow("Quantized Image", quantized)
cv2.waitKey(0)