from scipy.signal import wiener
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread('C:/Users/kirti/OneDrive/Desktop/image processing lab/im.jpeg', 0)

# Add sinusoidal noise
noise = 20 * np.sin(np.linspace(0, 50, img.shape[1]))
noise = noise.reshape(1, -1)

# Add noise to image
noisy = img + noise

# Clip values and convert to uint8
noisy = np.clip(noisy, 0, 255).astype(np.uint8)

# Apply Wiener filter
wiener_img = wiener(noisy)

# Display result
plt.imshow(wiener_img, cmap='gray')
plt.title('Wiener Filtered Image')
plt.axis('off')
plt.show()