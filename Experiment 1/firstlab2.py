import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\kirti\OneDrive\Desktop\im.jpeg")

if img is None:
    print("ERROR: Image not loaded. Check path or filename.")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Acquired Image")
    plt.axis("off")
    plt.show()
    print(img.shape)
    print(img[100,100])