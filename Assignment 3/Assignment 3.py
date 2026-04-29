import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def rle_encode(img):
    flat = img.flatten()
    pixels = flat.tolist()
    encoding = []
    if not pixels: return ""
    prev = pixels[0]
    count = 1
    for i in range(1, len(pixels)):
        if pixels[i] == prev:
            count += 1
        else:
            encoding.append((prev, count))
            prev = pixels[i]
            count = 1
    encoding.append((prev, count))
    return encoding

def run_medical_system(image_path=r"C:\Users\DELL\OneDrive\Desktop\python\sample 5.jpg"):
    print("="*60)
    print("MEDICAL IMAGE COMPRESSION & SEGMENTATION SYSTEM")
    print("="*60)
    
    img = cv2.imread(image_path, 0)
    if img is None:
        print("Error: Image not found.")
        return

    img = cv2.resize(img, (512, 512))
    
    encoded_data = rle_encode(img)
    original_size = img.size
    compressed_size = len(encoded_data) * 2
    ratio = original_size / compressed_size
    savings = (1 - (compressed_size / original_size)) * 100

    ret1, thresh_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret2, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5,5), np.uint8)
    morphed = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel)

    print(f"Compression Ratio: {ratio:.2f}")
    print(f"Storage Savings: {savings:.2f}%")

    images = [img, thresh_global, thresh_otsu, morphed]
    titles = ['Original Medical Image', 'Global Thresholding', 'Otsu Thresholding', 'Morphological Refinement']

    plt.figure(figsize=(12, 10))
    for i in range(len(images)):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/medical_results.png')
    plt.show()

if __name__ == "__main__":
    run_medical_system()