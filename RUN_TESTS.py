import torch
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from FINAL_UNET_MODEL import UNET

class TestDataset(Dataset):
    def __init__(self, image_dir, size=(256, 256)):
        self.image_dir = image_dir
        self.size = size
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        
        # Load the test image
        image = cv2.imread(os.path.join(self.image_dir, img_name))
        
        # Resize
        image = cv2.resize(image, self.size) / 255.0
        
        # Convert to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]

        return image, img_name
    

# Load the trained U-Net model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNET(in_channels=3, out_channels=1).to(device)  # Use 3 input channels (single image)
model.load_state_dict(torch.load("/home/cv-hacker/best_unet_model.pth", map_location=device))


model.eval()  # Set model to evaluation mode


# Define test dataset and DataLoader

test_dataset = TestDataset("/home/cv-hacker/test/test/images")  # Path to test images
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create output directory
os.makedirs("test_output_masks", exist_ok=True)

# Run inference and save masks
for image, img_name in test_loader:
    image = image.to(device)  # Add batch dimension
    
    with torch.no_grad():
        predicted_mask = model(image)
    
    # Convert mask to numpy and normalize
    predicted_mask = predicted_mask.squeeze().cpu().numpy()
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask
    
    # Save output mask
    output_path = os.path.join("test_output_masks", img_name[0])
    cv2.imwrite(output_path, predicted_mask)

    print(f"Saved mask: {output_path}")