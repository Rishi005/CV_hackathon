import os
import cv2
import numpy as np
import pandas as pd

def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Directory containing mask images
mask_dir = "/home/cv-hacker/test_output_masks"
out_csv = "/home/cv-hacker/masks_rle2.csv"

data = []

# Process each mask in the directory
for filename in sorted(os.listdir(mask_dir)):
    if filename.endswith(".png"):
        file_path = os.path.join(mask_dir, filename)
        
        # Load the mask image (grayscale)
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure mask is binary (255 -> 1)
        mask = (mask > 127).astype(np.uint8)
        
        # Convert to RLE
        rle = mask2rle(mask)

        filename_no_ext = os.path.splitext(filename)[0] 
        
        # Store in list
        data.append([filename_no_ext, rle])

# Create and save CSV
rle_df = pd.DataFrame(data, columns=["ImageId", "EncodedPixels"])
rle_df.to_csv(out_csv, index=False)

print(f"Saved RLE masks to {out_csv}")

