import os
import sys
import json
import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Ensure local imports operate properly matching the repository structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset import AIDetectionDataset
from utils.preprocess import get_transform
from models.model_loader import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train AI vs Real Image Classifier")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs)")
    parser.add_argument("--train-dir", type=str, default="dataset/train", help="Path to training data dir")
    parser.add_argument("--val-dir", type=str, default="dataset/val", help="Path to validation data dir")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory to save model weights and mapping")
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision implementation
        with autocast(enabled=device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({"Loss": loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Validation")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        with autocast(enabled=device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    args = parse_args()
    
    logger.info("Initializing Preprocessing Transform...")
    transform = get_transform(is_train=True)
    val_transform = get_transform(is_train=False)

    # 1. Boot up DataLoaders
    try:
        train_dataset = AIDetectionDataset(root_dir=args.train_dir, transform=transform)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=2 if os.name != 'nt' else 0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    except Exception as e:
        logger.error(f"Failed loading train directory: {e}. Make sure '{args.train_dir}/real' exists.")
        return
        
    try:
        val_dataset = AIDetectionDataset(root_dir=args.val_dir, transform=val_transform)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=2 if os.name != 'nt' else 0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    except Exception as e:
        logger.error(f"Failed loading val directory: {e}. Make sure '{args.val_dir}/real' exists.")
        return

    # 2. Save Class Configuration
    os.makedirs(args.save_dir, exist_ok=True)
    class_idx_path = os.path.join(args.save_dir, "class_to_idx.json")
    with open(class_idx_path, 'w') as f:
        json.dump(train_dataset.class_to_idx, f, indent=4)
    logger.info(f"Exported label configurations natively to {class_idx_path}")

    # 3. Inject Model loader capabilities
    logger.info("Booting up GPU mapped model architecture...")
    model_save_path = os.path.join(args.save_dir, "model.pth")
    model, device = get_model(model_save_path, num_classes=len(train_dataset.classes))

    # 4. Bind AI parameters - UPGRADED for accuracy
    # Label smoothing helps generalise better
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # AdamW is often superior to Adam for training deep networks
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # Cosine Annealing is a strong default scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = GradScaler(enabled=device.type == 'cuda')

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 5. Core Operational Loop
    logger.info(f"System ready! Initiating Core Phase for {args.epochs} cycles on {device}")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()

        logger.info(f"Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        logger.info(f"Val   -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # Early Stopping & Best Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Improved weights exported to disk with Val Loss @ {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    logger.info(f"Termination Complete! Model saved with best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
