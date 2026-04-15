import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from backend.app.core.config import MODEL_PATH


logger = logging.getLogger(__name__)

_MODEL_INSTANCE = None
_DEVICE_INSTANCE = None


def get_model(model_path: str | None = None, num_classes: int = 2):
    global _MODEL_INSTANCE, _DEVICE_INSTANCE

    if _MODEL_INSTANCE is not None and _DEVICE_INSTANCE is not None:
        return _MODEL_INSTANCE, _DEVICE_INSTANCE

    resolved_model_path = model_path or str(MODEL_PATH)

    _DEVICE_INSTANCE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)

    if Path(resolved_model_path).exists():
        try:
            state_dict = torch.load(resolved_model_path, map_location=_DEVICE_INSTANCE)
            model.load_state_dict(state_dict)
        except Exception as exc:
            logger.error("Failed to load weights from %s: %s", resolved_model_path, exc)
            logger.warning("Proceeding with initialized model weights.")
    else:
        logger.warning("Weight file not found at %s. Using initialized model.", resolved_model_path)

    _MODEL_INSTANCE = model.to(_DEVICE_INSTANCE)
    _MODEL_INSTANCE.eval()
    return _MODEL_INSTANCE, _DEVICE_INSTANCE
