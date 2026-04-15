from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import shutil
import tempfile

from models.model_loader import get_model
from models.inference import predict
from xai.gradcam import generate_heatmap
from xai.explanation import generate_explanation

# Instantiate System Event Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI Core Router
app = FastAPI(
    title="AI vs Real Vision Engine",
    description="Endpoint for dynamic generative synthesis visualization and diagnostics.",
    version="1.0.0"
)

# CORS Policy configuration for external Frontend UI bindings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bind Static Directory Routing (for public access to Heatmap Image elements)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

# Order matters: serve the heatmaps first
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# (Frontend mount moved to bottom to ensure API routes take precedence)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing GPU Model allocations and PyTorch singletons...")
    # This fires up the default model mapping architecture before any actual user requests execute
    get_model(model_path="model.pth", num_classes=2)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Vision ML Endpoint Operations"}

# @app.post("/analyze") is defined below


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Primary endpoint receiving uploaded graphic assets for inference mappings.
    Returns structurally integrated predictions, confidence heuristics and heatmaps.
    """
    logger.info(f"Incoming visual package payload received -> [{file.filename}]")
    
    # 1. Native extension limitation handling
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        logger.warning(f"Rejected unsupported file extension -> {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid visual envelope. Ensure payload is JPEG, PNG, or WEBP.")

    temp_path = None
    try:
        # 2. Spool Payload Memory File safely to local Disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
            
        logger.info(f"[Inference Thread] Image successfully cached locally at {temp_path}")

        # 3. ML Inference Pass
        prediction_result = predict(temp_path)
        
        if "error" in prediction_result:
            raise HTTPException(status_code=500, detail=prediction_result["error"])
            
        pred_label = prediction_result["prediction"]
        confidence = prediction_result["confidence"]

        # 4. Extract Explanatory Component Visuals (GradCAM computation)
        model, _ = get_model() # Instantly grabs context managed module through logic Singletons
        heatmap_full_path = generate_heatmap(temp_path, model, opacity=0.55)
        
        if "Error" in heatmap_full_path:
             raise HTTPException(status_code=500, detail=heatmap_full_path)
             
        # Extract native endpoint asset URL configuration
        heatmap_filename = os.path.basename(heatmap_full_path)
        heatmap_url = f"/static/heatmaps/{heatmap_filename}"
        
        # 5. Semantic Human Interpretation Processing
        explanation = generate_explanation(pred_label, confidence)
        
        logger.info(f"Analysis Thread Completed | Flagged: {pred_label} | Margin: {confidence}%")

        # 6. Structured Clean Output JSON Response
        return {
            "prediction": pred_label,
            "confidence": confidence,
            "explanation": explanation,
            "heatmap_url": heatmap_url
        }

    except HTTPException as http_e:
        # Standard HTTP rethrows
        raise http_e
    except Exception as e:
        logger.error(f"Catastrophic Thread Diagnostics Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Diagnostic Error: {str(e)}")
    finally:
        # Always shred operational tracking components regardless of failures or crash sequences!
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("[Thread Cleanup] Cached visual successfully deleted.")
            except Exception as cleanup_err:
                logger.warning(f"[Thread Cleanup] Failure to shred operational buffer: {cleanup_err}")

# Bind Frontend UI Routing (Built React Assets)
# IMPORTANT: This must be the VERY LAST mount to avoid capturing API routes like /analyze
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    logger.warning("Frontend 'dist' directory not found. Please run 'npm run build' in the frontend folder.")
