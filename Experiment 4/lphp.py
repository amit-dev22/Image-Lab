import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'C:\Users\kirti\OneDrive\Desktop\image processing lab\im.jpeg', 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = rows//2, cols//2

mask_lp = np.zeros((rows, cols), np.uint8)
mask_lp[crow-30:crow+30, ccol-30:ccol+30] = 1

mask_hp = 1 - mask_lp

lp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_lp))
hp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_hp))

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(lp), cmap='gray')
plt.title("Low Pass Filtered Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(hp), cmap='gray')
plt.title("High Pass Filtered Image")
plt.axis('off')
plt.show()