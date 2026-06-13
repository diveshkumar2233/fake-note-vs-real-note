import os
import pickle
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# ── Load .env ─────────────────────────────────────────────────────
load_dotenv()

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, os.getenv("MODEL_PATH",  "models/model.h5"))
SCALER_PATH = os.path.join(BASE_DIR, os.getenv("SCALER_PATH", "models/scaler.pkl"))

# ── Load model at startup ──────────────────────────────────────────
ml = {}   # shared state dict

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    from tensorflow.keras.models import load_model
    with open(SCALER_PATH, "rb") as f:
        ml["scaler"] = pickle.load(f)
    ml["model"] = load_model(MODEL_PATH)
    print(f"✅ Model  loaded → {MODEL_PATH}")
    print(f"✅ Scaler loaded → {SCALER_PATH}")
    yield
    # shutdown (nothing to clean up)
    ml.clear()

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="NoteGuard API",
    description="🔍 Fake vs Real Bank Note Detector — powered by TensorFlow + FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Static files & templates ───────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Pydantic schemas ───────────────────────────────────────────────
class NoteFeatures(BaseModel):
    variance: float = Field(..., example=3.6216,  description="Variance of wavelet transform")
    skewness: float = Field(..., example=8.6661,  description="Skewness of wavelet transform")
    curtosis: float = Field(..., example=-2.8073, description="Curtosis of wavelet transform")
    entropy:  float = Field(..., example=-0.4470, description="Entropy of image")

class PredictionResult(BaseModel):
    prediction: str   # "Real" or "Fake"
    confidence: float # 0.0 – 100.0


# ── Routes ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def index(request: Request):
    """Serve the main web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=PredictionResult, tags=["ML"])
async def predict(note: NoteFeatures):
    """
    **Authenticate a bank note.**

    Send the four wavelet-transformed image features and receive:
    - `prediction` → `"Real"` or `"Fake"`
    - `confidence` → percentage confidence (0–100)
    """
    if not ml:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    features = np.array([[
        note.variance,
        note.skewness,
        note.curtosis,
        note.entropy,
    ]])

    features_scaled = ml["scaler"].transform(features)
    raw             = ml["model"].predict(features_scaled)   # (1,1)
    prob_fake       = float(raw[0][0])
    is_fake         = prob_fake >= 0.5
    label           = "Fake" if is_fake else "Real"
    confidence      = prob_fake * 100 if is_fake else (1 - prob_fake) * 100

    return PredictionResult(prediction=label, confidence=round(confidence, 2))


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "app": os.getenv("APP_NAME", "NoteGuard"),
        "version": "1.0.0",
    }


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", 8000)),
        reload=os.getenv("FLASK_DEBUG", "1") == "1",
    )