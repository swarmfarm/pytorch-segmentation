#!/usr/bin/env python3
"""
Complete example demonstrating the KeyMakr segmentation dataset usage with PyTorch DataLoader.
This script shows how to:
1. Set up the dataset with proper transforms
2. Create train and validation dataloaders
3. Train a segmentation model using the KeyMakr dataset
4. Visualize predictions and class mappings

Run this script after adjusting the paths to your KeyMakr data.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# Add the parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from datasets.keymakr import KeymakrSegmentation, create_keymakr_dataloader
import transforms as T
from models import segmentation
import utils


def get_keymakr_transforms(train=True, resolution=(512, 512)):
    """
    Create transform pipeline for KeyMakr dataset.
    
    Args:
        train (bool): Whether to apply training augmentations
        resolution (tuple): Target resolution (height, width)
    
    Returns:
        Transform pipeline
    """
    transforms = []
    
    # Resize to target resolution
    transforms.append(T.Resize(resolution))
    
    # Apply data augmentation during training
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # Note: Be careful with random crops for segmentation - 
        # make sure masks and images are cropped consistently
    
    # Convert to tensors
    transforms.append(T.ToTensor())
    
    # Normalize using ImageNet statistics
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    
    return T.Compose(transforms)


def visualize_sample(dataset, index, save_path=None):
    """
    Visualize a sample from the dataset including image, mask, and class info.
    
    Args:
        dataset: KeyMakrSegmentation dataset instance
        index (int): Sample index to visualize
        save_path (str, optional): Path to save the visualization
    """
    # Get raw sample (without transforms for better visualization)
    dataset_no_transform = KeymakrSegmentation(
        root_dir=dataset.root_dir,
        image_set=dataset.image_set,
        transforms=None,  # No transforms for visualization
        val_split=dataset.val_split
    )
    
    image, mask = dataset_no_transform[index]
    
    # Convert to numpy arrays
    image_np = np.array(image)
    mask_np = np.array(mask)
    
    # Get class information
    class_info = dataset.get_class_info()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot segmentation mask
    im = axes[1].imshow(mask_np, cmap='tab20')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1])
    
    # Plot class distribution
    unique_classes, counts = np.unique(mask_np, return_counts=True)
    class_names = [class_info['index_to_class'].get(cls, f'Class_{cls}') for cls in unique_classes]
    
    axes[2].bar(range(len(unique_classes)), counts)
    axes[2].set_xticks(range(len(unique_classes)))
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].set_title('Class Distribution')
    axes[2].set_ylabel('Pixel Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_model(num_classes, arch='fcn_resnet18', pretrained=False):
    """
    Create a segmentation model.
    
    Args:
        num_classes (int): Number of output classes
        arch (str): Model architecture
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        PyTorch model
    """
    model = segmentation.__dict__[arch](
        num_classes=num_classes,
        aux_loss=None,
        pretrained=pretrained
    )
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Handle model output format
        if isinstance(outputs, dict):
            outputs = outputs['out']
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        num_samples += images.size(0)
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
    return avg_loss


def evaluate_model(model, dataloader, device, num_classes):
    """
    Evaluate the model on validation data.
    """
    model.eval()
    confusion_matrix = utils.ConfusionMatrix(num_classes)
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle model output format
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            # Get predictions
            predictions = outputs.argmax(1)
            
            # Update confusion matrix
            confusion_matrix.update(targets.flatten(), predictions.flatten())
    
    confusion_matrix.compute()
    print(f'Validation Results:')
    print(f'Mean IoU: {confusion_matrix.mean_IoU:.2f}%')
    print(f'Global Accuracy: {confusion_matrix.acc_global:.2f}%')
    
    return confusion_matrix


def main():
    parser = argparse.ArgumentParser(description='KeyMakr Segmentation Example')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to KeyMakr dataset directory')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--resolution', type=int, nargs=2, default=[512, 512],
                       help='Input resolution [height, width]')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Fraction of data for validation')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--model-arch', type=str, default='fcn_resnet18',
                       choices=['fcn_resnet18', 'fcn_resnet34', 'fcn_resnet50', 
                               'deeplabv3_resnet50', 'deeplabv3_resnet101'],
                       help='Model architecture')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create transforms
    train_transforms = get_keymakr_transforms(train=True, resolution=tuple(args.resolution))
    val_transforms = get_keymakr_transforms(train=False, resolution=tuple(args.resolution))
    
    # Create datasets and dataloaders
    print("Creating train dataloader...")
    train_dataloader, class_info = create_keymakr_dataloader(
        root_dir=args.data_dir,
        image_set='train',
        batch_size=args.batch_size,
        num_workers=4,
        transforms=train_transforms,
        val_split=args.val_split,
        random_seed=42
    )
    
    print("Creating validation dataloader...")
    val_dataloader, _ = create_keymakr_dataloader(
        root_dir=args.data_dir,
        image_set='val',
        batch_size=1,  # Use batch size 1 for validation
        num_workers=4,
        transforms=val_transforms,
        val_split=args.val_split,
        random_seed=42
    )
    
    # Print dataset information
    print(f"\nDataset Information:")
    print(f"Number of classes: {class_info['num_classes']}")
    print(f"Classes: {list(class_info['class_to_index'].keys())}")
    print(f"Train samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    
    # Visualize samples if requested
    if args.visualize:
        print("\nCreating visualizations...")
        
        # Create dataset without transforms for visualization
        viz_dataset = KeymakrSegmentation(
            root_dir=args.data_dir,
            image_set='train',
            transforms=None,
            val_split=args.val_split
        )
        
        if len(viz_dataset) > 0:
            visualize_sample(viz_dataset, 0, save_path='keymakr_sample_0.png')
            if len(viz_dataset) > 1:
                visualize_sample(viz_dataset, 1, save_path='keymakr_sample_1.png')
    
    # Create model
    print(f"\nCreating {args.model_arch} model...")
    model = create_model(class_info['num_classes'], arch=args.model_arch)
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        # Train
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, epoch + 1)
        
        # Validate
        confusion_matrix = evaluate_model(model, val_dataloader, device, class_info['num_classes'])
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'mean_iou': confusion_matrix.mean_IoU,
            'class_info': class_info,
            'args': args
        }
        
        torch.save(checkpoint, f'keymakr_checkpoint_epoch_{epoch + 1}.pth')
        print(f'Checkpoint saved: keymakr_checkpoint_epoch_{epoch + 1}.pth')
    
    print("\nTraining completed!")
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_confusion_matrix = evaluate_model(model, val_dataloader, device, class_info['num_classes'])
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_info': class_info,
        'args': args,
        'final_mean_iou': final_confusion_matrix.mean_IoU
    }, 'keymakr_final_model.pth')
    
    print("Final model saved: keymakr_final_model.pth")


if __name__ == '__main__':
    main()