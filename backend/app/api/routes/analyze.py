import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.schemas.analysis import AnalysisResponse
from backend.app.services.prediction_service import predict


logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid image type. Supported types: JPEG, PNG, WEBP.",
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)

        prediction_result = predict(temp_path)
        if "error" in prediction_result:
            raise HTTPException(status_code=500, detail=prediction_result["error"])

        logger.info(
            "Analysis completed for %s: %s (%s%%)",
            file.filename,
            prediction_result["prediction"],
            prediction_result["confidence"],
        )
        return prediction_result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected analysis failure")
        raise HTTPException(status_code=500, detail=f"Internal diagnostic error: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

