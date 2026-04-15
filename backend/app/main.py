import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes.analyze import router as analyze_router
from backend.app.api.routes.health import router as health_router
from backend.app.core.config import FRONTEND_DIST_DIR, MODEL_PATH
from backend.app.core.settings import get_allowed_origins
from backend.app.services.model_service import get_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI vs Real Vision Engine",
    description="Image classification API for AI-generated versus real images.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(analyze_router)


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing model")
    get_model(model_path=str(MODEL_PATH), num_classes=2)


if FRONTEND_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
else:
    logger.warning(
        "Frontend 'dist' directory not found. Run 'npm run build' in the frontend folder."
    )
