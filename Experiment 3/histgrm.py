import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'C:/Users/kirti/OneDrive/Desktop/image processing lab/im.jpeg', 0)
he_img = cv2.equalizeHist(img)
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1);plt.imshow(img, cmap='gray');plt.title("Original Image");plt.axis('off')
plt.subplot(1, 3, 2);plt.imshow(he_img, cmap='gray');plt.title("Contrast Stretched Image");plt.axis('off')
plt.subplot(1, 3, 3);plt.imshow(he_img, cmap='gray');plt.title("Histogram Equalized Image");plt.axis('off')
plt.show()