import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import logging

logger = logging.getLogger(__name__)

# Singleton variable caching
_MODEL_INSTANCE = None
_DEVICE_INSTANCE = None

def get_model(model_path: str = "model.pth", num_classes: int = 2):
    """
    Returns the EfficientNet model and device using a singleton pattern.
    Auto-detects GPU (CUDA) and falls back to CPU.
    Loads weights from model.pth if the file exists.
    
    Args:
        model_path (str): Path to the trained weights.
        num_classes (int): Number of output classes (e.g. 2 for real/ai).
        
    Returns:
        tuple: (model, device)
    """
    global _MODEL_INSTANCE, _DEVICE_INSTANCE
    
    # Return immediately if already loaded (Singleton Pattern)
    if _MODEL_INSTANCE is not None and _DEVICE_INSTANCE is not None:
        return _MODEL_INSTANCE, _DEVICE_INSTANCE
        
    logger.info("Initializing model loading process (Singleton initialization)...")
    
    # 1. GPU Auto-detection
    _DEVICE_INSTANCE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {_DEVICE_INSTANCE}")
    
    # 2. Architect the model (EfficientNet-B0)
    logger.info("Loading EfficientNet-B0 architecture...")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    # 3. Replace the final layer to output only our target classes
    # For EfficientNet, the classifier is usually `model.classifier[1]`
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
    
    # 4. Load our specifically trained weights if available
    if os.path.exists(model_path):
        logger.info(f"Loading custom weights from {model_path}...")
        try:
            # map_location ensures we handle CPU machines loading GPU-trained states safely
            state_dict = torch.load(model_path, map_location=_DEVICE_INSTANCE)
            model.load_state_dict(state_dict)
            logger.info("Weights loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load weights from {model_path}: {e}")
            logger.warning("Proceeding with initialized PyTorch model state.")
    else:
        logger.warning(f"Weight file '{model_path}' not found at path. Using initialized model.")
        
    # Move model to target device and force evaluation mode
    model = model.to(_DEVICE_INSTANCE)
    model.eval()
    
    _MODEL_INSTANCE = model
    
    logger.info("Model setup complete. Ready for inference requests.")
    return _MODEL_INSTANCE, _DEVICE_INSTANCE

# Preserved stub for existing main.py compatibility preventing any breakage
def load_model(model_path: str):
    """
    Placeholder wrapper to prevent breaking existing main.py structure.
    In the future this will be fully replaced by get_model().
    """
    logger.info(f"load_model wrapper called for '{model_path}'")
    # For now, just trigger our real loader logic without crashing the app flow
    return get_model(model_path)[0]
