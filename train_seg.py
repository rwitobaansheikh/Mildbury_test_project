import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
#from dataloader import SegmentationDataset
from model import UNet
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import csv
import time
from datetime import datetime
from PIL import Image

SEGMENTATION_COLOURS = {0:[0,0,0],1:[255,0,0],2:[0,253,0],3:[0,0,250], 4:[253,255,0]}

def calculate_iou(pred_mask, target_mask, num_classes=5):
    ious = []
    
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    if torch.is_tensor(target_mask):
        target_mask = target_mask.cpu().numpy()
    
    for cls in range(num_classes):
        pred_inds = pred_mask == cls
        target_inds = target_mask == cls
        
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        
        iou = intersection / (union + 1e-6)  
        ious.append(iou)
    
    return np.mean(ious)

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform

        self.image_dir = os.path.join(root_dir, 'Images')
        self.mask_dir = os.path.join(root_dir, 'Labels')
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for img in self.image_files:
            base_name = os.path.splitext(img)[0]
            mask_name = f"{base_name}_mask.png"
            mask_path = os.path.join(self.mask_dir, mask_name)


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        base_name = os.path.splitext(self.image_files[idx])[0]
        mask_name = os.path.join(self.mask_dir, f"{base_name}_mask.png")
        
        # Load image and mask
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('RGB')  # Keep as RGB for color mapping
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform_mask(mask)
        
        # Convert RGB mask to class indices
        mask_np = np.array(mask)
        class_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
        
        # Map colors to class indices
        for class_idx, color in SEGMENTATION_COLOURS.items():
            # Find all pixels that match this color
            matches = np.all(mask_np == np.array(color), axis=-1)
            class_mask[matches] = class_idx
        
        return image, torch.from_numpy(class_mask).long()
    
    def transform_mask(self, mask):
        """Apply only the spatial transforms to the mask (resize)"""
        if self.transform is None:
            return np.array(mask)
        
        # Get the resize transform if it exists
        for t in self.transform.transforms:
            if isinstance(t, transforms.Resize):
                mask = t(mask)
                break
        
        return np.array(mask)
    
def print_latest_metrics_table():
    metrics_dir = 'metrics'
    try:
        if not os.path.isdir(metrics_dir):
            print(f"Metrics directory '{metrics_dir}' not found. Run training first.")
            return

        all_files = [
            os.path.join(metrics_dir, f)
            for f in os.listdir(metrics_dir)
            if os.path.isfile(os.path.join(metrics_dir, f)) and
               f.startswith('training_metrics_') and f.endswith('.csv')
        ]
        if not all_files:
            print(f"No 'training_metrics_*.csv' files found in the '{metrics_dir}' directory.")
            return

        latest_file = max(all_files, key=os.path.getmtime)
        print(f"\n--- Training Metrics Summary from: {os.path.basename(latest_file)} ---")

        with open(latest_file, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader) # Read header row
                # Expected header: ['Epoch', 'Train Loss', 'Val Loss', 'Mean IoU', 'Time (s)']
                print(f"{'Epoch':<7} | {'Train Loss':<12} | {'Val Loss':<10} | {'Mean IoU':<10} | {'Time (s)':<10}")
                print("-" * 60) # Separator line

                for i, row in enumerate(reader):
                    if len(row) == 5:
                        try:
                            epoch = row[0]
                            train_loss = float(row[1])
                            val_loss = float(row[2])
                            mean_iou = float(row[3])
                            time_s = float(row[4])
                            print(f"{epoch:<7} | {train_loss:<12.4f} | {val_loss:<10.4f} | {mean_iou:<10.4f} | {time_s:<10.2f}")
                        except ValueError:
                            print(f"Warning: Could not parse row {i+1} in {os.path.basename(latest_file)}: {row}")
                    else:
                        print(f"Warning: Skipping malformed row {i+1} in {os.path.basename(latest_file)} (expected 5 columns): {row}")
            except StopIteration:
                print(f"Warning: The metrics file {os.path.basename(latest_file)} is empty or has no data rows.")

    except FileNotFoundError:
        print(f"Error: The directory '{metrics_dir}' or a specific metrics file was not found. Ensure training has run.")
    except Exception as e:
        print(f"An error occurred while trying to print metrics: {e}")
        
def print_latest_metrics_table():
    metrics_dir = 'metrics'
    try:
        if not os.path.isdir(metrics_dir):
            print(f"Metrics directory '{metrics_dir}' not found. Run training first.")
            return

        all_files = [
            os.path.join(metrics_dir, f)
            for f in os.listdir(metrics_dir)
            if os.path.isfile(os.path.join(metrics_dir, f)) and
               f.startswith('training_metrics_') and f.endswith('.csv')
        ]
        if not all_files:
            print(f"No 'training_metrics_*.csv' files found in the '{metrics_dir}' directory.")
            return

        latest_file = max(all_files, key=os.path.getmtime)
        print(f"\n--- Training Metrics Summary from: {os.path.basename(latest_file)} ---")

        with open(latest_file, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader) # Read header row
                # Expected header: ['Epoch', 'Train Loss', 'Val Loss', 'Mean IoU', 'Time (s)']
                print(f"{'Epoch':<7} | {'Train Loss':<12} | {'Val Loss':<10} | {'Mean IoU':<10} | {'Time (s)':<10}")
                print("-" * 60) # Separator line

                for i, row in enumerate(reader):
                    if len(row) == 5:
                        try:
                            epoch = row[0]
                            train_loss = float(row[1])
                            val_loss = float(row[2])
                            mean_iou = float(row[3])
                            time_s = float(row[4])
                            print(f"{epoch:<7} | {train_loss:<12.4f} | {val_loss:<10.4f} | {mean_iou:<10.4f} | {time_s:<10.2f}")
                        except ValueError:
                            print(f"Warning: Could not parse row {i+1} in {os.path.basename(latest_file)}: {row}")
                    else:
                        print(f"Warning: Skipping malformed row {i+1} in {os.path.basename(latest_file)} (expected 5 columns): {row}")
            except StopIteration:
                print(f"Warning: The metrics file {os.path.basename(latest_file)} is empty or has no data rows.")

    except FileNotFoundError:
        print(f"Error: The directory '{metrics_dir}' or a specific metrics file was not found. Ensure training has run.")
    except Exception as e:
        print(f"An error occurred while trying to print metrics: {e}")
        
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    
    metrics_dir = 'metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_file = os.path.join(metrics_dir, f'training_metrics_{timestamp}.csv')
    
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Mean IoU', 'Time (s)'])
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': train_loss / len(train_loader)})
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mean_iou = calculate_iou(all_preds, all_targets)
        
        epoch_time = time.time() - epoch_start_time        
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, mean_iou, epoch_time])
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Mean IoU: {mean_iou:.4f}')
        print(f'  Time: {epoch_time:.2f}s')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SegmentationDataset('dataset/train', transform=transform)
    val_dataset = SegmentationDataset('dataset/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = UNet(n_classes=5).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    print_latest_metrics_table()

if __name__ == '__main__':
    main() 
    