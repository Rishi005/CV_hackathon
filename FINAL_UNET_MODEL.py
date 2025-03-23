import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


#unet model
class DoubleConv(nn.Module):
   def __init__(self, in_channels, out_channels):
       super(DoubleConv, self).__init__()
       self.conv = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace=False),
           nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),  # Fixed from your original code
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace=False),           
       )


   def forward(self, x):
       return self.conv(x)
  
class UNET(nn.Module):
   def __init__(
           self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]
   ):
       super(UNET, self).__init__()
       self.ups = nn.ModuleList()
       self.downs = nn.ModuleList()
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      
       #encode
       for feature in features:
           self.downs.append(DoubleConv(in_channels, feature))
           in_channels = feature
    
       #decode
       for feature in reversed(features):
           self.ups.append(
               nn.ConvTranspose2d(
                   feature*2, feature, kernel_size=2, stride=2
               )
           )
           self.ups.append(DoubleConv(feature*2, feature))




       # Bottleneck
       self.bottleneck = DoubleConv(features[-1], features[-1]*2)
      
       # final convolution sigmoid + binary segmentation
       self.final_conv = nn.Sequential(
           nn.Conv2d(features[0], out_channels, kernel_size=1),
           nn.Sigmoid()  # Added sigmoid for binary output
       )


   def forward(self, x):
       print(f"Input to UNET: {x.shape}")
       skip_connections = []


       for down in self.downs:
           x = down(x)
           print(f"After down block: {x.shape}")
           skip_connections.append(x)
           x = self.pool(x)
           print(f"After pooling: {x.shape}")


       x = self.bottleneck(x)
       print(f"After bottleneck: {x.shape}")
       skip_connections = skip_connections[::-1]


       for idx in range(0, len(self.ups), 2):
           x = self.ups[idx](x)
           print(f"After transpose conv: {x.shape}")
           skip_connection = skip_connections[idx//2]
           print(f"Skip connection: {skip_connection.shape}")
      
           if x.shape != skip_connection.shape:
               x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
               print(f"After resize: {x.shape}")  # Debug
          
           concat_skip = torch.cat((skip_connection, x), dim=1)
           print(f"After concat: {concat_skip.shape}")
           x = self.ups[idx+1](concat_skip)
           print(f"After conv: {x.shape}")

       return self.final_conv(x)


class ManipulationDataset(Dataset):
   def __init__(self, image_dir, mask_dir, transform=None):
       self.image_dir = image_dir
       self.mask_dir = mask_dir
       self.transform = transform
       self.images = [file for file in os.listdir(image_dir) if file.endswith('.png')]
  
   def __len__(self):
       return len(self.images)
  
   def __getitem__(self, idx):
       img_path = os.path.join(self.image_dir, self.images[idx])
       mask_path = os.path.join(self.mask_dir, self.images[idx])
  
       # Read image and mask
       image = cv2.imread(img_path)
       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
  
       # Read mask as grayscale
       mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
  
       # Ensure mask is binary (0 or 1)
       mask = mask / 255.0 if mask.max() > 1 else mask
  
   # Apply transformations to both image and mask
       if self.transform:
           augmented = self.transform(image=image, mask=mask)
           image = augmented['image'].float()  # Ensure float32
           mask = augmented['mask'].float()    # Ensure float32
      
       # Ensure mask has channel dimension
           if len(mask.shape) == 2:
               mask = mask.unsqueeze(0)  # Add channel dimension: [1, H, W]
       else:
       # Manual conversion with explicit float()
           image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0  # [C, H, W]
           mask = torch.from_numpy(mask).float().unsqueeze(0)  # [1, H, W]

       return image, mask


# Comprehensive augmentation functions
def get_training_augmentation(height=256, width=256):
   return A.Compose([
        A.Resize(height=height, width=width),

        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    ], p=0.7),

       # Saturation and color modifications
       A.OneOf([
           A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
           A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
           A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
           A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.7),
       ], p=0.4),
       A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       ToTensorV2(), 
], p=1.0)


def get_validation_augmentation(height=256, width=256):
   return A.Compose([
       A.Resize(height=height, width=width),
       A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       ToTensorV2(), 
   ], p=1.0)


def visualize_augmentations(dataset, idx=0, samples=5):
   """Visualize augmentations on a sample from the dataset"""
   image, mask = dataset[idx]
  
   # Convert tensor to numpy for OpenCV
   if isinstance(image, torch.Tensor):
       image = image.permute(1, 2, 0).numpy()
       image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
       image = image.astype(np.uint8)
  
   if isinstance(mask, torch.Tensor):
       mask = mask.squeeze().numpy()
  
   original_img = cv2.imread(os.path.join(dataset.image_dir, dataset.images[idx]))
   original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
  
   original_mask = cv2.imread(os.path.join(dataset.mask_dir, dataset.images[idx]), cv2.IMREAD_GRAYSCALE)
   original_mask = original_mask / 255.0 if original_mask.max() > 1 else original_mask
  
   fig, axes = plt.subplots(samples + 1, 2, figsize=(10, (samples + 1) * 5))
       # Create validation dataset with validation transforms
   # Display original
   axes[0, 0].imshow(original_img)
   axes[0, 0].set_title("Original Image")
   axes[0, 0].axis('off')
  
   axes[0, 1].imshow(original_mask, cmap='gray')
   axes[0, 1].set_title("Original Mask")
   axes[0, 1].axis('off')
  
   # Display augmented samples
   for i in range(samples):
       augmented_sample = dataset[idx]
       aug_img = augmented_sample[0]
       aug_mask = augmented_sample[1]
      
       # Convert tensor to numpy for visualization
       if isinstance(aug_img, torch.Tensor):
           aug_img = aug_img.permute(1, 2, 0).numpy()
           aug_img = (aug_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
           aug_img = aug_img.astype(np.uint8)
      
       if isinstance(aug_mask, torch.Tensor):
           aug_mask = aug_mask.squeeze().numpy()
      
       axes[i+1, 0].imshow(aug_img)
       axes[i+1, 0].set_title(f"Augmented Image {i+1}")
       axes[i+1, 0].axis('off')
      
       axes[i+1, 1].imshow(aug_mask, cmap='gray')
       axes[i+1, 1].set_title(f"Augmented Mask {i+1}")
       axes[i+1, 1].axis('off')
  
   plt.tight_layout()
   plt.show()


def dice_coefficient(pred, target, smooth=1e-6):
   """Calculate Dice coefficient"""
   pred = pred.view(-1)
   target = target.view(-1)
   intersection = (pred * target).sum()
   union = pred.sum() + target.sum()
   dice = (2.0 * intersection + smooth) / (union + smooth)
   return dice


def dice_loss(pred, target, smooth=1e-6):
   """Dice loss function"""
   return 1 - dice_coefficient(pred, target, smooth)

class BCEDiceLoss(nn.Module):
   """Combined BCE and Dice loss"""
   def __init__(self, weight_bce=0.5, weight_dice=0.5):
       super(BCEDiceLoss, self).__init__()
       self.weight_bce = weight_bce
       self.weight_dice = weight_dice
      
   def forward(self, pred, target):
       # Ensure both inputs are float32
       pred = pred.float()
       target = target.float()
      
       bce_loss = nn.BCELoss()(pred, target)
       dice_l = dice_loss(pred, target)
      
       return self.weight_bce * bce_loss + self.weight_dice * dice_l


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda'):
   # Debug: Check shapes before training
   sample_image, sample_mask = next(iter(train_loader))
   print(f"Batch shape - images: {sample_image.shape}, masks: {sample_mask.shape}")
   """Train the U-Net model"""
   best_val_dice = 0.0
   history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
  
   model = model.to(device)
  
   for epoch in range(num_epochs):
       model.train()
       train_loss = 0.0
       train_samples = 0
      
       train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
       for images, masks in train_loop:
           images = images.to(device)
           masks = masks.to(device)
          
           # Forward pass
           optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, masks)
          
           # Backward pass
           loss.backward()
           optimizer.step()
          
           # Update stats
           train_loss += loss.item() * images.size(0)
           train_samples += images.size(0)
          
           train_loop.set_postfix(loss=train_loss/train_samples)
      
       avg_train_loss = train_loss / train_samples
       history['train_loss'].append(avg_train_loss)
      
       # Validation phase
       model.eval()
       val_loss = 0.0
       val_dice = 0.0
       val_samples = 0
      
       val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
       with torch.no_grad():
           for images, masks in val_loop:
               images = images.to(device)
               masks = masks.to(device)
              
               outputs = model(images)
               loss = criterion(outputs, masks)
              
               # Calculate metrics
               dice = dice_coefficient(outputs, masks).item()
              
               val_loss += loss.item() * images.size(0)
               val_dice += dice * images.size(0)
               val_samples += images.size(0)
              
               val_loop.set_postfix(loss=val_loss/val_samples, dice=val_dice/val_samples)
      
       avg_val_loss = val_loss / val_samples
       avg_val_dice = val_dice / val_samples
      
       history['val_loss'].append(avg_val_loss)
       history['val_dice'].append(avg_val_dice)
      
       print(f"Epoch {epoch+1}/{num_epochs}")
       print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
      
       # Save best model
       if avg_val_dice > best_val_dice:
           best_val_dice = avg_val_dice
           torch.save(model.state_dict(), 'best_unet_model.pth')
           print(f"Saved new best model with Dice score: {best_val_dice:.4f}")
  
   return model, history

def main():
   image_dir = '/home/cv-hacker/dataset/train/train/images'
   mask_dir = '/home/cv-hacker/dataset/train/train/masks'


   train_transform = get_training_augmentation()
   train_dataset = ManipulationDataset(
       image_dir=image_dir,
       mask_dir=mask_dir,
       transform=train_transform
   )


   val_transform = get_validation_augmentation()
   val_dataset = ManipulationDataset(
       image_dir=image_dir,
       mask_dir=mask_dir,
       transform=val_transform
   )


   val_size = int(0.1 * len(train_dataset))
   train_size = len(train_dataset) - val_size
  
   indices = list(range(len(train_dataset)))
   np.random.shuffle(indices)
   train_indices = indices[:train_size]
   val_indices = indices[train_size:]


   train_set = torch.utils.data.Subset(train_dataset, train_indices)
   val_set = torch.utils.data.Subset(val_dataset, val_indices)

   train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
   val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
  

   for images, masks in train_loader:
       print(f"Images shape: {images.shape}") 
       print(f"Masks shape: {masks.shape}")   
       break
  


   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = UNET(in_channels=3, out_channels=1)
   criterion = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)
   optimizer = optim.Adam(model.parameters(), lr=1e-4)
  
 #trainig
   model, history = train_model(
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       criterion=criterion,
       optimizer=optimizer,
       num_epochs=25,
       device=device
   )

   plt.figure(figsize=(12, 5))
  
   plt.subplot(1, 2, 1)
   plt.plot(history['train_loss'], label='Train Loss')
   plt.plot(history['val_loss'], label='Val Loss')
   plt.title('Loss')
   plt.xlabel('Epoch')
   plt.legend()
  
   plt.subplot(1, 2, 2)
   plt.plot(history['val_dice'], label='Val Dice')
   plt.title('Dice Coefficient')
   plt.xlabel('Epoch')
   plt.legend()
  
   plt.tight_layout()
   plt.savefig('training_history.png')
   plt.show()


if __name__ == "__main__":
   main()


