import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'C:/Users/kirti/OneDrive/Desktop/image processing lab/im.jpeg', 0)

min_val, max_val = np.min(img), np.max(img)

cs_img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
plt.imshow(cs_img, cmap='gray')
plt.title("Contrast Stretched Image")
plt.axis('off')
plt.show()