import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def run_restoration(image_path = r"C:\Users\DELL\OneDrive\Desktop\python\sample 4.jpg"):

    print("="*60)
    print("SURVEILLANCE IMAGE RESTORATION SYSTEM")
    print("="*60)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find image at {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("[Success] Image loaded and converted to grayscale.")

    print("[Processing] Adding simulated surveillance noise...")
    
    row, col = gray.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col)).astype('uint8')
    noisy_gaussian = cv2.add(gray, gauss)

    # 2. Salt-and-Pepper Noise (Transmission Error) [cite: 21]
    noisy_sp = gray.copy()
    prob = 0.05
    thres = 1 - prob
    for i in range(row):
        for j in range(col):
            rdn = np.random.random()
            if rdn < prob:
                noisy_sp[i][j] = 0
            elif rdn > thres:
                noisy_sp[i][j] = 255

    # Task 3: Image Restoration Techniques [cite: 23-28]
    print("[Processing] Applying spatial filters for restoration...")
    
    # Mean Filter [cite: 25]
    mean_filtered = cv2.blur(noisy_gaussian, (5, 5))
    
    # Median Filter (Best for S&P noise) [cite: 26, 36]
    median_filtered = cv2.medianBlur(noisy_sp, 5)
    
    # Gaussian Filter [cite: 27]
    gaussian_filtered = cv2.GaussianBlur(noisy_gaussian, (5, 5), 0)

    # Task 4: Performance Evaluation (MSE & PSNR) [cite: 29-33]
    def get_metrics(orig, restored):
        mse = np.mean((orig - restored) ** 2)
        psnr = cv2.PSNR(orig, restored)
        return mse, psnr

    mse_m, psnr_m = get_metrics(gray, median_filtered)   
    print("\n" + "-"*30)
    print("FILTER PERFORMANCE COMPARISON:")
    print(f"Median Filter (on S&P Noise) -> MSE: {mse_m:.2f}, PSNR: {psnr_m:.2f} dB")
    print("\nJUSTIFICATION:")
    print("- Median filter is superior for Salt-and-Pepper noise as it removes outliers.") [cite: 36, 37]
    print("- Mean/Gaussian filters are better for Gaussian (sensor) noise but blur edges.")
    print("-"*30)
    titles = ['Original', 'Gaussian Noise', 'S&P Noise', 'Mean Filter', 'Median Filter', 'Gaussian Filter']
    images = [gray, noisy_gaussian, noisy_sp, mean_filtered, median_filtered, gaussian_filtered]

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/restoration_results.png')
    print("\n[Final] Output saved to outputs/restoration_results.png")
    plt.show()

if __name__ == "__main__":
    run_restoration()