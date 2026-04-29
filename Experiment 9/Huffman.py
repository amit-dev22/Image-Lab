import cv2
import numpy as np

# Read image in grayscale
img = cv2.imread(r'C:\Users\kirti\OneDrive\Desktop\KRMU\image processing lab\im.jpeg', 0)

# Check if image loaded correctly
if img is None:
    print("Error: Image not found or path is incorrect.")
    exit()

flat = img.flatten()

# Run Length Encoding
def rle_encode(data):
    if len(data) == 0:
        return []

    encoding = []
    prev = data[0]
    count = 1

    for i in data[1:]:
        if i == prev:
            count += 1
        else:
            encoding.append((prev, count))
            prev = i
            count = 1

    encoding.append((prev, count))
    return encoding

encoded = rle_encode(flat)

print("Original size:", len(flat))
print("Encoded size:", len(encoded))