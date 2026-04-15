from backend.app.services.model_service import get_model


def load_model(model_path: str):
    return get_model(model_path)[0]

