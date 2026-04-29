import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def run_traffic_system(image_path=r"C:\Users\DELL\OneDrive\Desktop\python\sample 6.jpg"):
    print("="*60)
    print("FEATURE-BASED TRAFFIC MONITORING SYSTEM")
    print("="*60)

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    canny = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()
    
    print("\nDetected Objects Data:")
    for i, cnt in enumerate(contours[:5]):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"Object {i+1}: Area={area:.2f}, Perimeter={perimeter:.2f}")

    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(gray, None)
    feature_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))

    images = [sobel_combined, canny, contour_img, feature_img]
    titles = ['Sobel Edge Detection', 'Canny Edge Detection', 'Contours & Bounding Boxes', 'ORB Feature Extraction']

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(2, 2, i+1)
        if i < 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/traffic_results.png')
    plt.show()

if __name__ == "__main__":
    run_traffic_system()