import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def process_scanner(image_path="C:\\Users\\DELL\\OneDrive\\Desktop\\python\\sample 1.jpeg"):
    # Task 1: Project Introduction [cite: 247]
    print("="*60)
    print("SYSTEM START: SMART DOCUMENT SCANNER & QUALITY ANALYSIS")
    print("="*60)
    print(f"Acquiring image from: {image_path}")

    # Task 2: Image Acquisition & Preprocessing [cite: 249-251]
    img = cv2.imread(image_path)
    if img is None: 
        print("Error: Image not found. Please check the file path.")
        return
    
    img = cv2.resize(img, (512, 512)) # Standardize resolution [cite: 250]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale conversion [cite: 251]
    print("[1/3] Image acquisition and grayscale conversion complete.")
    
    # Task 3: Image Sampling (Resolution Analysis) [cite: 253-258]
    print("[2/3] Performing resolution sampling analysis...")
    sampling_results = []
    # Simulating High (512), Medium (256), and Low (128) resolutions [cite: 255-257]
    for res in [512, 256, 128]:
        down = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA)
        up = cv2.resize(down, (512, 512), interpolation=cv2.INTER_NEAREST)
        sampling_results.append(up)
        
    # Task 4: Image Quantization (Gray Level Reduction) [cite: 260-264]
    print("[3/3] Performing bit-depth quantization analysis...")
    quant_results = []
    # Simulating 8-bit (256), 4-bit (16), and 2-bit (4) gray levels [cite: 262-264]
    for levels in [256, 16, 4]:
        factor = 256 // levels
        quantized = (gray // factor) * factor
        quant_results.append(quantized)
        
    # Task 5: Quality Observation & Analysis [cite: 266-270]
    print("\n" + "-"*30)
    print("ANALYTICAL OBSERVATIONS:")
    print("- Sampling: Lower resolutions (128x128) cause significant text blur and aliasing.")
    print("- Quantization: 2-bit (4 levels) causes heavy contouring and loss of fine text details.")
    print("- OCR Suitability: High-res 8-bit images are required for reliable text recognition.")
    print("-"*30)

    # Visualization and Final Output [cite: 271-276]
    titles = ['Original Grayscale', 'Res: 512x512', 'Res: 256x256', 'Res: 128x128', 
              'Quant: 8-bit', 'Quant: 4-bit', 'Quant: 2-bit']
    images = [gray] + sampling_results + quant_results
    
    plt.figure(figsize=(15, 8))
    plt.suptitle("Document Scanner Analysis: Sampling and Quantization Effects", fontsize=14)
    for i in range(len(images)):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    # Save output images to folder [cite: 274]
    os.makedirs('outputs', exist_ok=True)
    output_filename = 'outputs/scanner_comparison.png'
    plt.savefig(output_filename)
    print(f"\nAnalysis complete. Results saved in '{output_filename}'.")
    plt.show()

# Execution
if __name__ == "__main__":
    process_scanner()
