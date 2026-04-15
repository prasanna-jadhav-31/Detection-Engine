import os
from pathlib import Path
from typing import List, Tuple, Callable
import logging

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# Import preprocessing
import sys
if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess import get_transform

logger = logging.getLogger(__name__)

class AIDetectionDataset(Dataset):
    """
    Custom PyTorch Dataset for loading 'real' and 'ai' images.
    Expects a directory structure where `root_dir` is the split folder (e.g., train/ or val/):
        root_dir/
            real/
            ai/
    """
    def __init__(self, root_dir: str, transform: Callable = None):
        """
        Args:
            root_dir (str): Path to the split directory (e.g., 'dataset/train' or 'dataset/val').
            transform (Callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Define classes and mapping
        self.classes = ['real', 'ai']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            logger.warning(f"No valid images found in {self.root_dir}.")

    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        Scans directory for images and returns a list of (image_path, class_index) tuples.
        """
        samples = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            if not class_dir.exists() or not class_dir.is_dir():
                logger.warning(f"Directory not found - {class_dir}")
                continue
                
            for file_name in os.listdir(class_dir):
                file_path = class_dir / file_name
                if file_path.suffix.lower() in valid_extensions:
                    samples.append((str(file_path), class_idx))
                    
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image_tensor, label)
            If the image is corrupted, it skips the image and loads another one to avoid breaking training.
        """
        while True:
            image_path, label = self.samples[idx]
            try:
                # 1. Load image and immediately ensure it is in RGB format.
                # `convert('RGB')` handles grayscale and RGBA fallback easily ensuring 3 channels.
                image = Image.open(image_path).convert('RGB')
                
                # 2. Verify image isn't corrupted or truncated (raises Exception if broken)
                image.verify()
                
                # `verify()` can close the underlying file or modify the pointer, reopen:
                image = Image.open(image_path).convert('RGB')
                
                # 3. Apply transformations
                if self.transform is not None:
                    image_tensor = self.transform(image)
                else:
                    # Fallback to transform to prevent errors returning a PIL Image when tensor expected
                    image_tensor = transforms.ToTensor()(image) 
                    
                return image_tensor, label
                
            except Exception as e:
                logger.warning(f"Skipping corrupted/unreadable image at {image_path}: {e}")
                # If reading fails, just pick the next index
                idx = (idx + 1) % max(len(self.samples), 1)
                
                # If the dataset is empty, we must break to avoid infinite loop
                if len(self.samples) == 0:
                    raise RuntimeError("Dataset is empty or contains no valid images.")


# --- Example Usage Snippet ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Setting up the Dataset and Preprocessing Pipeline...")
    
    # 1. Create a dummy dataset directory structure and images for demonstration
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        train_dir = Path(temp_dir) / 'dataset' / 'train'
        (train_dir / 'real').mkdir(parents=True, exist_ok=True)
        (train_dir / 'ai').mkdir(parents=True, exist_ok=True)
        
        # Create an RGB image (simulating real photo)
        Image.new('RGB', (300, 300), color='green').save(train_dir / 'real' / 'real_01.jpg')
        # Create a Grayscale image (simulating a BW AI generated image)
        Image.new('L', (400, 400), color=128).save(train_dir / 'ai' / 'ai_01.png')
        # Create a dummy corrupted file to test the fault-tolerance
        with open(train_dir / 'real' / 'real_corrupted.jpg', 'w') as f:
            f.write("I am not an image file")
            
        print(f"\nCreated mock dataset at {train_dir}")
        
        # 2. Load the transformations
        transform = get_transform(is_train=True)
        
        # 3. Initialize the PyTorch Dataset
        train_dataset = AIDetectionDataset(root_dir=str(train_dir), transform=transform)
        print(f"Loaded dataset with {len(train_dataset)} samples.")
        
        # 4. Use DataLoader for batching
        dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        
        # Fetch batches 
        for images, labels in dataloader:
            print(f"\nBatch Information:")
            print(f"- Images shape: {images.shape}")
            print(f"- Labels: {labels}")
            
            assert images.shape[1] == 3, "Failed: Images should have 3 channels (Grayscale -> RGB conversion failed)"
            assert images.shape[2] == 224 and images.shape[3] == 224, "Failed: Images were not resized to 224x224"
            print("\nVerification Successful: 3 Channels, Resized correctly, Corrupted image safely skipped!")
            break
