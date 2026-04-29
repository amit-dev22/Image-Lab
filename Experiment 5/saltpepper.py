import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread(r'C:/Users/kirti/OneDrive/Desktop/image processing lab/im.jpeg', 0)
# Gaussian Noise
gauss = img.astype(np.float32) + np.random.normal(0, 20, img.shape)
gauss = np.clip(gauss, 0, 255).astype(np.uint8)
# Salt & Pepper Noise
sp = img.copy()
prob = 0.02
rand = np.random.rand(*img.shape)
sp[rand < prob/2] = 0
sp[rand > 1 - prob/2] = 255
# Filters
mean_filt = cv2.blur(gauss, (3, 3))
median_filt = cv2.medianBlur(sp, 3)
# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(gauss, cmap='gray')
plt.title("Gaussian Noise Image")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(mean_filt, cmap='gray')
plt.title("Mean Filtered Image")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(median_filt, cmap='gray')
plt.title("Median Filtered Image")
plt.axis('off')

plt.show()