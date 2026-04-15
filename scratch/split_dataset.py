import os
import shutil
import random
from pathlib import Path

def split_dataset(base_dir="dataset", split_ratio=0.2):
    """
    Moves 20% of images from train to val to ensure a proper validation split.
    Also cleans up existing files in val/ to ensure no mock data pollution.
    """
    base_path = Path(base_dir)
    classes = ["real", "ai"]
    
    for cls in classes:
        train_dir = base_path / "train" / cls
        val_dir = base_path / "val" / cls
        
        # 1. Clean up val directory
        print(f"Cleaning up {val_dir}...")
        if val_dir.exists():
            for f in val_dir.iterdir():
                if f.is_file():
                    f.unlink()
        else:
            val_dir.mkdir(parents=True, exist_ok=True)
            
        # 2. Get list of files in train
        files = [f for f in train_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        num_to_move = int(len(files) * split_ratio)
        
        print(f"Found {len(files)} images in {train_dir}. Moving {num_to_move} to validation.")
        
        # 3. Randomly select and move
        to_move = random.sample(files, num_to_move)
        for f in to_move:
            shutil.move(str(f), str(val_dir / f.name))
            
    print("\nDataset rebalancing complete!")
    print(f"New counts:")
    for cls in classes:
        t_count = len(list((base_path / "train" / cls).iterdir()))
        v_count = len(list((base_path / "val" / cls).iterdir()))
        print(f"- {cls.capitalize()}: Train={t_count}, Val={v_count}")

if __name__ == "__main__":
    split_dataset()
