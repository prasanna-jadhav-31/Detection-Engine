import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from backend.app.core.config import CLASS_MAPPING_PATH, MODEL_PATH
from backend.app.services.model_service import get_model
from utils.preprocess import get_transform


def load_class_map() -> dict[int, str]:
    class_map = {0: "Real", 1: "AI"}
    if not CLASS_MAPPING_PATH.exists():
        return class_map

    with CLASS_MAPPING_PATH.open("r", encoding="utf-8") as file:
        class_to_idx = json.load(file)

    return {int(index): str(label).capitalize() for label, index in class_to_idx.items()}


def predict(image_path: str | Path) -> dict[str, str | float]:
    image_path = Path(image_path)
    if not image_path.exists():
        return {"error": f"Image file not found at {image_path}"}

    try:
        class_map = load_class_map()
        model, device = get_model(model_path=str(MODEL_PATH), num_classes=len(class_map))

        transform = get_transform(is_train=False)
        image = Image.open(image_path).convert("RGB")
        batch_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            max_confidence, predicted_index = torch.max(probabilities, 1)

        confidence = round(float(max_confidence.item()) * 100.0, 2)
        prediction = class_map.get(int(predicted_index.item()), "Unknown")
        return {"prediction": prediction, "confidence": confidence}
    except Exception as exc:
        return {"error": f"Inference execution failed: {exc}"}

