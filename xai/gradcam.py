import os
import cv2
import uuid
import numpy as np
import torch
import sys
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    # Explicitly catch lack of PIP package natively resolving missing components
    raise ImportError("Missing required dependency: pytorch-grad-cam. Please run: pip install grad-cam")

# Align internal environmental pathing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import get_transform

def generate_heatmap(image_path_or_model, model_or_data=None, opacity=0.5):
    """
    Generates and saves a Grad-CAM heatmap visualization overlay.
    
    Args:
        image_path (str): Path to original inference image.
        model (torch.nn.Module): Active PyTorch model instance natively loaded.
        opacity (float): Transparency index (0.0 to 1.0) governing the heatmap visibility overlay factor.
        
    Returns:
        str: Absolute or relative output file route natively pointing to /static/heatmaps/ where graphic saved.
    """
    # ====== Legacy Shim Wrapper ======
    if not isinstance(image_path_or_model, str):
        # Support fallback if accidentally pinged by older architecture styles 
        # (originally model, input_data signature)
        return []

    image_path = image_path_or_model
    model = model_or_data
    
    if model is None:
        return "Error: Missing active AI model injection."

    if not os.path.exists(image_path):
        return f"Error: Image {image_path} not found"

    # Prepare specific static output architecture pipeline
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "heatmaps")
    os.makedirs(out_dir, exist_ok=True)
    
    # Formulate unique tracking collision graphic filename
    filename = f"heatmap_{uuid.uuid4().hex[:8]}.jpg"
    heatmap_path = os.path.join(out_dir, filename)

    try:
        # 1. Isolate the target visualization layer
        # For Torchvision EfficientNet configurations, `.features[-1]` safely intercepts the final 
        # distinct convolutional map projection just before Pooling and Classifier sequences occur.
        target_layers = [model.features[-1]]
        
        # 2. Extract standard internal image metrics
        transform = get_transform(is_train=False) 
        raw_image = Image.open(image_path).convert("RGB")
        img_tensor = transform(raw_image)
        
        # Attach required computational batching vector B=1 -> [1, C, H, W]
        batch_tensor = img_tensor.unsqueeze(0)
        
        # Auto-align native hardware detection from architecture origin
        device = next(model.parameters()).device
        input_tensor = batch_tensor.to(device)
        
        # Initialize Core GradCAM API Context Manager 
        # targets=None triggers tracking automatically on the most highly-activated output sequence
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            
        # Isolate scalar mapping vectors 
        grayscale_cam = grayscale_cam[0, :]

        # 3. Canvas Resizing Operations (Map original input visually to bounded shape mappings)
        img_canvas = cv2.imread(image_path)
        img_canvas = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2RGB)
        
        # Both image canvas native shape and tensor spatial properties must align visually perfectly
        base_size = (img_tensor.shape[1], img_tensor.shape[2]) # (224, 224) standard natively 
        img_canvas = cv2.resize(img_canvas, base_size)
        
        # Matrix Scaling limit bound map => Float mapping safely within [0,1] graphic boundaries
        rgb_img = np.float32(img_canvas) / 255.0

        # 4. Integrate computational heatmaps onto raw input
        # Note: image_weight dictates background pixel intensity mapped directly out of inverse input 1.0 - opacity 
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=1.0 - opacity)
        
        # 5. Flush directly to hard disk utilizing BGR cvt properties
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_path, overlay_bgr)
        
        return heatmap_path

    except Exception as e:
        return f"Error during Grad-CAM visualization pipeline extraction: {str(e)}"
