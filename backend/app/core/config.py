from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "model.pth"
CLASS_MAPPING_PATH = BASE_DIR / "class_to_idx.json"
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"

