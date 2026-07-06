"""
main.py
-------
FastAPI backend for the Volleyball Pose Analysis web app.

Architecture recap:
    - The BROWSER does all the heavy lifting: camera capture, MediaPipe
      pose detection, skeleton drawing, and joint-angle math all run
      client-side in JavaScript (see templates/index.html).
    - The SERVER only receives a tiny JSON payload of two floats
      (elbow_angle, knee_angle) and returns a classification instantly
      using a pre-trained scikit-learn SVM. This keeps the network
      payload minimal and the server workload trivial, which is exactly
      what we need on a free/CPU-Basic Hugging Face Space.

Endpoints:
    GET  /         -> serves the single-page frontend (templates/index.html)
    POST /predict  -> accepts {elbow_angle, knee_angle}, returns
                       {form_label, feedback}
    GET  /health   -> simple health check, useful for HF Spaces / uptime checks
"""

import os
import subprocess
import sys

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Volleyball Pose Analysis API",
    description="Real-time spike-form classification from elbow/knee joint angles.",
    version="1.0.0",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "app_model.joblib")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Make sure the static directory exists before mounting it (it may be empty,
# but StaticFiles requires the directory to physically exist).
os.makedirs(STATIC_DIR, exist_ok=True)

# Serve any future assets (icons, extra CSS/JS) placed in /static under /static.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Jinja2 is used purely to serve the single index.html template (no dynamic
# templating variables are strictly required, but this keeps the door open
# for injecting server-side config into the page later).
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Holds the in-memory SVM model once loaded at startup.
model = None


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PoseFeatures(BaseModel):
    """Payload sent by the browser for every throttled prediction request."""

    elbow_angle: float = Field(..., ge=0, le=180, description="Right elbow angle in degrees")
    knee_angle: float = Field(..., ge=0, le=180, description="Right knee angle in degrees")


class PredictionResponse(BaseModel):
    form_label: int
    feedback: str


# ---------------------------------------------------------------------------
# Startup: load the pre-trained model (train it on the fly as a fallback)
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_model() -> None:
    """Load 'app_model.joblib' from disk. If it's missing (e.g. a fresh
    checkout without having run the Docker build step), fall back to
    running train_svm.py right now so the server can still start and
    serve predictions."""
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"[startup] '{MODEL_PATH}' not found. Running fallback training routine...")
        train_script = os.path.join(BASE_DIR, "train_svm.py")
        subprocess.run([sys.executable, train_script], check=True, cwd=BASE_DIR)

    model = joblib.load(MODEL_PATH)
    print("[startup] SVM model loaded successfully.")


# ---------------------------------------------------------------------------
# Feedback generation
# ---------------------------------------------------------------------------
def build_feedback(form_label: int, elbow_angle: float, knee_angle: float) -> str:
    """Translate the raw SVM label into a short, actionable coaching tip
    based on which joint is furthest from the ideal range."""

    if form_label == 1:
        return "Excellent spike form! Full arm extension and a well-loaded knee for maximum power."

    # form_label == 0 -> figure out which joint needs the most correction
    # so the athlete gets a specific, actionable cue instead of a generic one.
    elbow_ideal_mid = 165.0
    knee_ideal_mid = 117.0

    elbow_error = abs(elbow_angle - elbow_ideal_mid)
    knee_error = abs(knee_angle - knee_ideal_mid)

    tips = []
    if elbow_angle < 145:
        tips.append("extend your hitting arm fully at contact (snap the elbow straighter)")
    if knee_angle > 140:
        tips.append("bend your knees more to load your legs before the jump")
    if knee_angle < 95:
        tips.append("avoid over-crouching; ease up on the knee bend slightly")

    if tips:
        return "Needs adjustment: " + " and ".join(tips) + "."

    # Fallback generic message if neither threshold was clearly triggered
    # (e.g. borderline values the SVM still classified as 0).
    dominant_joint = "elbow" if elbow_error > knee_error else "knee"
    return f"Needs adjustment: focus on your {dominant_joint} positioning for better spike mechanics."


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serve the single-page frontend that runs the camera + MediaPipe pipeline."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    """Lightweight health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PoseFeatures):
    """Classify a single (elbow_angle, knee_angle) reading using the SVM.

    Returns:
        form_label: 1 = Good Spike Form, 0 = Needs Adjustment
        feedback:   a short human-readable coaching tip
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please retry shortly.")

    # Build the feature vector in the exact same shape used during training:
    # a single row with [elbow_angle, knee_angle].
    X = np.array([[features.elbow_angle, features.knee_angle]])

    form_label = int(model.predict(X)[0])
    feedback = build_feedback(form_label, features.elbow_angle, features.knee_angle)

    return PredictionResponse(form_label=form_label, feedback=feedback)


# ---------------------------------------------------------------------------
# Entrypoint for local development (Hugging Face Spaces uses the Dockerfile's
# CMD instead, but this makes `python main.py` work too for quick testing).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
