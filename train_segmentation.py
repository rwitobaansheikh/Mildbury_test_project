import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloader import SegmentationDataset
from model import UNet
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import time
from datetime import datetime

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

if __name__ == '__main__':
    main() 