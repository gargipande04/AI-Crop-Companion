"""
Crop Companion: FastAPI Backend

Run with:
    python main.py
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.config import ASSETS_DIR, TEMPLATES_DIR
from app.services.ai_chatbot import chat
from app.services.disease_detection import diagnose, disease_model_status, load_disease_model
from app.services.pest_detection import identify_pest, load_pest_model, pest_model_status
from app.services.yield_engine import get_metadata, get_yield_model_names, predict_yield


@asynccontextmanager
async def lifespan(app):
    """
    FastAPI lifespan context manager.

    Model loading is intentionally not done here. Loading torch models
    inside an async context caused deadlocks on macOS because the event
    loop executor and torch's internal thread pool conflicted. Models are
    loaded synchronously in the __main__ block instead, before uvicorn
    starts its event loop.
    """
    yield


app = FastAPI(title="Crop Companion API", lifespan=lifespan)

# Allow all origins so the browser frontend can reach the API during local
# development without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictInput(BaseModel):
    """Request body for the /predict endpoint."""

    area: str
    crop: str
    rainfall: float
    temperature: float
    pesticides: float
    income_level: Optional[str] = "medium"


@app.get("/metadata")
def metadata():
    """
    Return all data needed to populate the frontend form dropdowns and
    display the Model Comparison panel.
    """
    return get_metadata()


@app.post("/predict")
def predict(data: PredictInput):
    """Run a yield prediction and generate a sustainability report."""
    return predict_yield(data)


app.post("/diagnose")(diagnose)
app.get("/disease-model-status")(disease_model_status)
app.post("/identify-pest")(identify_pest)
app.get("/pest-model-status")(pest_model_status)
app.post("/chat")(chat)


@app.get("/health")
def health():
    """
    Health check endpoint. Returns the status of all loaded components.
    Used by the frontend loading screen to verify the backend is ready.
    """
    return {
        "status": "ok",
        "disease_model_loaded": disease_model_status()["loaded"],
        "yield_models": get_yield_model_names(),
    }


# Mount the assets directory as /static so the frontend can access
# images and JavaScript via /static/<subdir>/<filename>.
app.mount("/static", StaticFiles(directory=str(ASSETS_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serve the research landing page (home.html) at the root URL."""
    html_path = TEMPLATES_DIR / "home.html"
    if not html_path.exists():
        return HTMLResponse("<h2>home.html not found</h2>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/app", response_class=HTMLResponse)
def serve_frontend():
    """Serve the main application UI (index.html) at /app."""
    html_path = TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h2>index.html not found</h2>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


def run() -> None:
    """
    Start the application after synchronously loading optional torch models.
    """
    import uvicorn

    load_disease_model()
    load_pest_model()

    print("\n" + "=" * 50)
    print("Crop Companion API is ready")
    print(f"  Yield models : {get_yield_model_names()}")
    print(f"  Disease model: {'loaded' if disease_model_status()['loaded'] else 'not found'}")
    print(f"  Pest model   : {'loaded' if pest_model_status()['loaded'] else 'not found — run train_pest_model.py'}")
    print("  Open: http://127.0.0.1:8000")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
