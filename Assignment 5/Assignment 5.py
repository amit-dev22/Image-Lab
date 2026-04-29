import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def run_capstone_system(image_path=r"C:\Users\DELL\OneDrive\Desktop\python\sample 7.jpg"):
    print("="*60)
    print("INTELLIGENT IMAGE ENHANCEMENT & ANALYSIS SYSTEM")
    print("="*60)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find image at {image_path}")
        return
    
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("[1/5] Image acquired and preprocessed.")

    row, col = gray.shape
    gauss_noise = np.random.normal(0, 20, (row, col)).astype('uint8')
    noisy = cv2.add(gray, gauss_noise)
    restored = cv2.medianBlur(noisy, 5)
    enhanced = cv2.equalizeHist(restored)
    print("[2/5] Noise restoration and enhancement complete.")

    _, segmented = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    morphed = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, kernel)
    print("[3/5] Segmentation and morphological refinement complete.")

    canny = cv2.Canny(morphed, 100, 200)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    features = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
    print("[4/5] Edge detection and feature extraction complete.")

    mse = np.mean((gray - restored) ** 2)
    psnr = cv2.PSNR(gray, restored)
    
    print("\n" + "-"*30)
    print("SYSTEM PERFORMANCE METRICS:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
    
    if HAS_SKIMAGE:
        score, _ = ssim(gray, restored, full=True)
        print(f"Structural Similarity Index (SSIM): {score:.4f}")
    else:
        print("SSIM: [Module Missing - Run 'pip install scikit-image' to enable]")
    print("-" * 30)

    display_imgs = [gray, noisy, restored, enhanced, morphed, features]
    display_titles = ['Original', 'Noisy', 'Restored', 'Enhanced', 'Segmented', 'Features']

    plt.figure(figsize=(15, 10))
    for i in range(len(display_imgs)):
        plt.subplot(2, 3, i+1)
        if i == 5:
            plt.imshow(cv2.cvtColor(display_imgs[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(display_imgs[i], cmap='gray')
        plt.title(display_titles[i])
        plt.axis('off')

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/capstone_pipeline.png')
    print("\n[5/5] Success! Visual pipeline saved to 'outputs/capstone_pipeline.png'")
    plt.show()

if __name__ == "__main__":
    run_capstone_system()
