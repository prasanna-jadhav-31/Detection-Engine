import os
import torch
from PIL import Image
import numpy as np

def generate_mock_images(base_dir="dataset", num_per_class=10):
    for split in ["train", "val"]:
        for cls in ["real", "ai"]:
            dir_path = os.path.join(base_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)
            
            for i in range(num_per_class):
                # Random color image
                color = tuple(np.random.randint(0, 255, 3))
                img = Image.new("RGB", (224, 224), color=color)
                
                # Add some random noise to make them slightly different
                data = np.array(img)
                noise = np.random.randint(-20, 20, data.shape)
                data = np.clip(data + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(data)
                
                img.save(os.path.join(dir_path, f"{cls}_{i:03d}.jpg"))
                
    print(f"Generated {num_per_class} mock images for each class in train and val splits.")

if __name__ == "__main__":
    generate_mock_images()
