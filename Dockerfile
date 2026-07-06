# ==============================================================================
# Dockerfile - Volleyball Pose Analysis (Hugging Face Spaces / Docker SDK)
#
# Design goals:
#   - Small, fast image based on python:3.10-slim (CPU Basic tier friendly).
#   - The SVM model is trained ONCE at build time (train_svm.py), so the
#     container starts instantly with 'app_model.joblib' already on disk -
#     no training work happens on every cold start.
#   - Hugging Face Spaces runs containers as a non-root user with UID 1000,
#     so all app files are made readable/writable by that user.
#   - Listens on port 7860, which is the default port HF Spaces expects.
# ==============================================================================

FROM python:3.10-slim

# Avoid .pyc files and force stdout/stderr to be unbuffered (cleaner logs
# in the HF Spaces log viewer).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- System-level dependencies -----------------------------------------------
# Only minimal build tools are needed for scikit-learn's compiled wheels;
# most modern wheels are prebuilt, but gcc is kept as a safety net for
# platforms without a matching prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ------------------------------------------------------
# Copied and installed before the rest of the app so Docker can cache this
# layer across rebuilds that only touch application code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code ----------------------------------------------------------
COPY main.py .
COPY train_svm.py .
COPY templates/ ./templates/
COPY static/ ./static/

# --- Train the SVM model at build time ----------------------------------------
# This guarantees 'app_model.joblib' exists in the image before the server
# ever starts, so the very first request is served by a ready model.
RUN python train_svm.py

# --- Permissions for Hugging Face Spaces' non-root runtime user (UID 1000) ---
# HF Spaces executes the container as user 1000, so make sure that user can
# read/write everything under /app (e.g. if the model ever needs retraining
# or logs need to be written at runtime).
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Hugging Face Spaces expects the app to listen on port 7860.
EXPOSE 7860

# --- Start the FastAPI server via Uvicorn -------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
