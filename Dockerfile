FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    MODEL_ID=shi-labs/oneformer_coco_swin_large \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache \
    HF_HUB_CACHE=/opt/hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCH_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    FORCE_CPU=1 \
    MAX_SIDE=512 \
    MAX_UPLOAD_MB=10 \
    MIN_AREA_PX=25 \
    OVERLAY_ALPHA=0.55 \
    MAX_BOXES=50

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download the model and processor during image build.
# This prevents Cloud Run from downloading model files into the runtime writable filesystem.
RUN mkdir -p /opt/hf_cache && python - <<'PY'
import os
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

model_id = os.environ.get("MODEL_ID", "shi-labs/oneformer_coco_swin_large")
print(f"Pre-downloading model during Docker build: {model_id}", flush=True)

OneFormerProcessor.from_pretrained(model_id)
OneFormerForUniversalSegmentation.from_pretrained(model_id)

print("Model and processor cached successfully.", flush=True)
PY

# After the model is cached in the image, prevent unexpected online downloads at runtime.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

COPY main.py .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
