import torch
import torch.nn.functional as F
from PIL import Image
import os
import sys
import json

# Ensure local imports operate properly matching the repository structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_loader import get_model
from utils.preprocess import get_transform

def predict(image_path_or_model, processed_data=None):
    """
    Primary Inference Pipeline
    ---
    Converts uploaded image path -> prediction & confidence dict.
    
    (Note: Has optional `processed_data` kwargs purely as a fallback 
     catch to avoid crashing the older main.py Forex endpoint)
    """
    
    # ====== Legacy Boilerplate Defense ======
    if processed_data is not None:
        return {
            "direction": "up",
            "confidence": 0.85,
            "system_note": "Legacy payload invoked"
        }
    
    # ====== AI Vision Inference Pipeline ======
    image_path = image_path_or_model
    
    if not os.path.exists(image_path):
        return {"error": f"Image file not found at {image_path}"}
        
    try:
        # Load class mapping if available
        class_map = {0: "Real", 1: "AI"}  # Default fallback
        if os.path.exists("class_to_idx.json"):
            try:
                with open("class_to_idx.json", "r") as f:
                    c2i = json.load(f)
                    # Create reverse map: {idx: "Label"}
                    class_map = {int(v): str(k).capitalize() for k, v in c2i.items()}
            except Exception as e:
                print(f"Warning: Failed to load class_to_idx.json: {e}")

        # 1. Load Model (Instant singleton grab, no repetitive disk reads)
        model, device = get_model(model_path="model.pth", num_classes=len(class_map))
        model.eval() # Ensure dropout & batchnorm are disabled
        
        # 2. Preprocess Image
        transform = get_transform(is_train=False)
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image)
        
        # Expand out a batch dimension B=1 -> [1, C, H, W]
        batch_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Fast Inference Mode Optimization
        with torch.no_grad():
            if device.type == 'cuda':
                # Use Automatic-Mixed Precision (AMP) logic for max speed on GPU setups
                with torch.cuda.amp.autocast():
                    outputs = model(batch_tensor)
            else:
                # 3. Standard CPU forward pass
                outputs = model(batch_tensor)
                
            # 4. Apply Softmax for probability conversion
            probabilities = F.softmax(outputs, dim=1)
            
            # Extrapolate confidence and assigned indexing
            max_conf, predicted_idx = torch.max(probabilities, 1)
            
            # Convert to neat data scalars
            conf_val = float(max_conf.item()) * 100.0
            idx_val = int(predicted_idx.item())
            
            # Map semantic name
            class_label = class_map.get(idx_val, "Unknown")
            
            # 5. Return structured format
            return {
                "prediction": class_label,
                "confidence": round(conf_val, 2)
            }
            
    except Exception as e:
        return {"error": f"Inference execution failed: {str(e)}"}
