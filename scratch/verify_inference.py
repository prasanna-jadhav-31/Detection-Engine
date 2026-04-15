import os
import sys
from PIL import Image
import numpy as np

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from models.inference import predict

def test_inference():
    # 1. Create a dummy test image
    test_img_path = "test_image.jpg"
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    img.save(test_img_path)
    
    print(f"Testing inference on {test_img_path}...")
    
    # 2. Run prediction
    result = predict(test_img_path)
    
    print("\nInference Result:")
    print(f"Prediction: {result.get('prediction')}")
    print(f"Confidence: {result.get('confidence')}%")
    if result.get('original_label'):
        print(f"Original Label: {result.get('original_label')}")
    
    # Clean up
    if os.path.exists(test_img_path):
        os.remove(test_img_path)

if __name__ == "__main__":
    test_inference()
