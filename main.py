"""
FastAPI service for OneFormer panoptic segmentation and object counting.

Designed for Google Cloud Run:
- The web server starts quickly on Cloud Run. The model is loaded lazily on the first analysis request.
- The API accepts either multipart image upload or base64 JSON.
- The response returns area-by-label, thing-object counts, instance boxes, and
  optionally a base64 overlay image with segmentation + bounding boxes.

Endpoints:
- GET  /health
- POST /analyze          multipart/form-data: file=<image>
- POST /analyze-image    same as /analyze
- POST /segment          same as /analyze
- POST /analyze-base64   application/json: {"image_base64": "..."}
"""

from __future__ import annotations

import base64
import io
import os
import random
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, Field
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor


# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = os.getenv("MODEL_ID", "shi-labs/oneformer_coco_swin_large")
DEFAULT_MAX_SIDE = int(os.getenv("MAX_SIDE", "1600"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "25"))
DEFAULT_MIN_AREA_PX = int(os.getenv("MIN_AREA_PX", "25"))
DEFAULT_OVERLAY_ALPHA = float(os.getenv("OVERLAY_ALPHA", "0.55"))
DEFAULT_MAX_BOXES = int(os.getenv("MAX_BOXES", "150"))

# Optional: limit CPU thread use in Cloud Run to avoid over-allocating threads.
CPU_THREADS = int(os.getenv("TORCH_NUM_THREADS", "1"))
torch.set_num_threads(max(1, CPU_THREADS))

# Use locks because this model is large. MODEL_LOAD_LOCK prevents two requests
# from loading the model at the same time. INFERENCE_LOCK prevents two heavy
# inferences from running simultaneously inside the same container.
MODEL_LOAD_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()


# =============================================================================
# COCO "thing" classes: used as a fallback when the model segment metadata does
# not explicitly mark an item as an object-like instance.
# =============================================================================

COCO_THING_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
}


# =============================================================================
# Model state
# =============================================================================

class ModelState:
    processor: Optional[OneFormerProcessor] = None
    model: Optional[OneFormerForUniversalSegmentation] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    loaded: bool = False


state = ModelState()


def load_model_once() -> None:
    """Load the processor and model once per container instance.

    Important for Cloud Run:
    This function is intentionally NOT called during FastAPI startup. Cloud Run
    first needs the web server to start and listen on PORT=8080. The model is
    therefore loaded lazily when the first /analyze request is received.
    """
    if state.loaded:
        return

    with MODEL_LOAD_LOCK:
        if state.loaded:
            return

        state.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model lazily: {MODEL_ID}", flush=True)
        print(f"Device: {state.device}", flush=True)

        state.processor = OneFormerProcessor.from_pretrained(MODEL_ID)
        state.model = OneFormerForUniversalSegmentation.from_pretrained(MODEL_ID)
        state.model.to(state.device)
        state.model.eval()

        state.loaded = True
        print("Model loaded successfully.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start FastAPI quickly so Cloud Run can mark the container as ready."""
    print(
        "Cloud Run container started. Model will be loaded lazily on the first analysis request.",
        flush=True,
    )
    yield


app = FastAPI(
    title="Urban Image Segmentation API",
    description="OneFormer panoptic segmentation, object counting, area-by-label reporting, and overlay generation.",
    version="1.0.0",
    lifespan=lifespan,
)


# CORS: during testing "*" is convenient. For production, set ALLOWED_ORIGINS
# to your front-end domain, for example:
# ALLOWED_ORIGINS=https://your-app.web.app,https://your-domain.com
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# =============================================================================
# Helper functions
# =============================================================================

def normalize_label(name: str) -> str:
    """Normalize model labels for robust object/stuff classification."""
    n = name.lower().replace("_", " ").replace("-", " ").strip()
    n = n.replace(" other merged", "").replace(" merged", "").strip()
    return n


def label_name_from_segment(seg: Dict[str, Any], id2label: Dict[int, str]) -> str:
    """Resolve a readable label for a panoptic segment."""
    if "label" in seg and isinstance(seg["label"], str):
        return seg["label"]

    idx = seg.get("label_id", seg.get("category_id"))
    if idx is not None:
        return id2label.get(int(idx), f"label_{idx}")

    return f"segment_{seg.get('id', '?')}"


def is_thing(seg: Dict[str, Any], label_name: str) -> bool:
    """
    Decide whether a segment is a countable object instance.
    The model usually marks this in metadata, but the COCO thing-list fallback
    is kept to match your Colab logic.
    """
    if bool(seg.get("isthing", False)) or bool(seg.get("is_thing", False)):
        return True
    return normalize_label(label_name) in COCO_THING_LABELS


def resize_if_needed(image: Image.Image, max_side: int) -> Tuple[Image.Image, bool, Dict[str, int]]:
    """Downscale large images to control memory and response time."""
    original_w, original_h = image.size
    if max(original_w, original_h) <= max_side:
        return image, False, {"original_width": original_w, "original_height": original_h}

    scale = max_side / float(max(original_w, original_h))
    new_size = (int(original_w * scale), int(original_h * scale))
    image = image.resize(new_size, Image.LANCZOS)
    return image, True, {
        "original_width": original_w,
        "original_height": original_h,
        "resized_width": new_size[0],
        "resized_height": new_size[1],
    }


def color_from_id(idx: int) -> Tuple[int, int, int]:
    """Stable pseudo-random color for each segment id."""
    rng = random.Random(int(idx))
    return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))


def gather_instances(
    segments_info: List[Dict[str, Any]],
    seg_np: np.ndarray,
    id2label: Dict[int, str],
    min_area_px: int = DEFAULT_MIN_AREA_PX,
) -> List[Dict[str, Any]]:
    """Extract segment area, object/stuff type, and bounding boxes."""
    instances: List[Dict[str, Any]] = []

    for seg in segments_info:
        seg_id = int(seg["id"])
        label = label_name_from_segment(seg, id2label)
        mask = seg_np == seg_id
        area = int(mask.sum())

        if area < min_area_px:
            continue

        rows, cols = np.where(mask)
        if rows.size == 0 or cols.size == 0:
            continue

        y_min, y_max = int(rows.min()), int(rows.max())
        x_min, x_max = int(cols.min()), int(cols.max())

        thing = is_thing(seg, label)
        instances.append(
            {
                "id": seg_id,
                "label": label,
                "normalized_label": normalize_label(label),
                "is_thing": thing,
                "area_px": area,
                "bbox_xyxy": [x_min, y_min, x_max, y_max],
                "score": seg.get("score", None),
            }
        )

    return instances


def area_by_label(
    segments_info: List[Dict[str, Any]],
    seg_np: np.ndarray,
    id2label: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Aggregate segmented pixel area by semantic label."""
    total_pixels = int(seg_np.size)
    by_label: Dict[str, int] = {}

    for seg in segments_info:
        seg_id = int(seg["id"])
        label = label_name_from_segment(seg, id2label)
        area = int((seg_np == seg_id).sum())
        by_label[label] = by_label.get(label, 0) + area

    result = []
    for label, area in sorted(by_label.items(), key=lambda item: item[1], reverse=True):
        result.append(
            {
                "label": label,
                "area_px": area,
                "percent_of_image": round((area / total_pixels) * 100.0, 2),
            }
        )
    return result


def count_things_by_label(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate countable object instances by label."""
    counts: Dict[str, int] = {}
    for inst in instances:
        if inst["is_thing"]:
            label = inst["label"]
            counts[label] = counts.get(label, 0) + 1

    return [
        {"label": label, "count": count}
        for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]


def make_overlay_image(
    image: Image.Image,
    seg_np: np.ndarray,
    segments_info: List[Dict[str, Any]],
    instances: List[Dict[str, Any]],
    alpha: float = DEFAULT_OVERLAY_ALPHA,
    max_boxes: int = DEFAULT_MAX_BOXES,
) -> Image.Image:
    """Create overlay image with panoptic segmentation and boxes for thing instances."""
    h, w = seg_np.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    for seg in segments_info:
        seg_id = int(seg["id"])
        overlay[seg_np == seg_id] = color_from_id(seg_id)

    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    alpha = max(0.0, min(1.0, float(alpha)))
    blended = (alpha * overlay + (1.0 - alpha) * base).astype(np.uint8)

    out = Image.fromarray(blended)
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()

    thing_instances = sorted(
        [inst for inst in instances if inst["is_thing"]],
        key=lambda item: item["area_px"],
        reverse=True,
    )[: max(0, int(max_boxes))]

    for inst in thing_instances:
        x1, y1, x2, y2 = inst["bbox_xyxy"]
        label = str(inst["label"])

        # White rectangle, black label background, white text.
        draw.rectangle([x1, y1, x2, y2], outline="white", width=2)

        try:
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except Exception:
            text_w = 8 * len(label)
            text_h = 12

        label_y1 = max(0, y1 - text_h - 4)
        draw.rectangle(
            [x1, label_y1, x1 + text_w + 6, label_y1 + text_h + 4],
            fill="black",
        )
        draw.text((x1 + 3, label_y1 + 2), label, fill="white", font=font)

    return out


def pil_to_base64_png(image: Image.Image) -> str:
    """Encode a PIL image as base64 PNG."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def open_image_from_bytes(raw: bytes) -> Image.Image:
    """Open uploaded bytes as an RGB PIL image with EXIF orientation corrected."""
    try:
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the uploaded image: {exc}") from exc


def run_segmentation(
    image: Image.Image,
    *,
    filename: str = "uploaded_image",
    max_side: int = DEFAULT_MAX_SIDE,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    return_overlay: bool = True,
    overlay_alpha: float = DEFAULT_OVERLAY_ALPHA,
    max_boxes: int = DEFAULT_MAX_BOXES,
) -> Dict[str, Any]:
    """Main analytical pipeline."""
    if not state.loaded:
        load_model_once()

    if state.processor is None or state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    max_side = max(256, min(int(max_side), 4096))
    min_area_px = max(1, int(min_area_px))

    image, resized, size_info = resize_if_needed(image, max_side=max_side)
    width, height = image.size

    with INFERENCE_LOCK:
        with torch.inference_mode():
            inputs = state.processor(
                images=image,
                task_inputs=["panoptic"],
                return_tensors="pt",
            )
            inputs = {
                key: (value.to(state.device) if hasattr(value, "to") else value)
                for key, value in inputs.items()
            }
            outputs = state.model(**inputs)

            panoptic = state.processor.post_process_panoptic_segmentation(
                outputs,
                target_sizes=[image.size[::-1]],  # (height, width)
            )[0]

    segmentation_map = panoptic["segmentation"]
    segments_info = panoptic["segments_info"]

    if isinstance(segmentation_map, torch.Tensor):
        seg_np = segmentation_map.detach().cpu().numpy().astype(np.int32)
    else:
        seg_np = np.asarray(segmentation_map).astype(np.int32)

    id2label = state.model.config.id2label
    instances = gather_instances(
        segments_info=segments_info,
        seg_np=seg_np,
        id2label=id2label,
        min_area_px=min_area_px,
    )

    areas = area_by_label(segments_info=segments_info, seg_np=seg_np, id2label=id2label)
    object_counts = count_things_by_label(instances)

    overlay_base64 = None
    if return_overlay:
        overlay = make_overlay_image(
            image=image,
            seg_np=seg_np,
            segments_info=segments_info,
            instances=instances,
            alpha=overlay_alpha,
            max_boxes=max_boxes,
        )
        overlay_base64 = pil_to_base64_png(overlay)

    response = {
        "success": True,
        "model": {
            "model_id": MODEL_ID,
            "device": state.device,
            "task": "panoptic",
        },
        "image": {
            "filename": filename,
            "width": width,
            "height": height,
            "total_pixels": int(width * height),
            "resized": resized,
            **size_info,
        },
        "settings": {
            "max_side": max_side,
            "min_area_px": min_area_px,
            "return_overlay": return_overlay,
            "overlay_alpha": overlay_alpha,
            "max_boxes": max_boxes,
        },
        "summary": {
            "number_of_segments": len(segments_info),
            "number_of_object_instances": sum(item["count"] for item in object_counts),
            "number_of_area_labels": len(areas),
        },
        "area_by_label": areas,
        "object_counts": object_counts,
        "instances": instances,
        "overlay_png_base64": overlay_base64,
    }
    return response


# =============================================================================
# API schemas and routes
# =============================================================================

class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description="Raw base64 PNG/JPG string or a data URL.")
    filename: str = "uploaded_image"
    max_side: int = DEFAULT_MAX_SIDE
    min_area_px: int = DEFAULT_MIN_AREA_PX
    return_overlay: bool = True
    overlay_alpha: float = DEFAULT_OVERLAY_ALPHA
    max_boxes: int = DEFAULT_MAX_BOXES


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "Urban Image Segmentation API",
        "status": "running",
        "model_loaded": state.loaded,
        "model_id": MODEL_ID,
        "device": state.device,
        "endpoints": {
            "health": "GET /health",
            "multipart_upload": "POST /analyze or POST /analyze-image",
            "base64_json": "POST /analyze-base64",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": state.loaded,
        "model_id": MODEL_ID,
        "device": state.device,
    }


@app.post("/analyze")
@app.post("/analyze-image")
@app.post("/segment")
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    max_side: int = Form(DEFAULT_MAX_SIDE),
    min_area_px: int = Form(DEFAULT_MIN_AREA_PX),
    return_overlay: bool = Form(True),
    overlay_alpha: float = Form(DEFAULT_OVERLAY_ALPHA),
    max_boxes: int = Form(DEFAULT_MAX_BOXES),
) -> Dict[str, Any]:
    """Analyze an image sent as multipart/form-data."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. Please upload an image file.",
        )

    raw = await file.read()
    max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Maximum upload size is {MAX_UPLOAD_MB:g} MB.",
        )

    image = open_image_from_bytes(raw)

    return run_segmentation(
        image,
        filename=file.filename or "uploaded_image",
        max_side=max_side,
        min_area_px=min_area_px,
        return_overlay=return_overlay,
        overlay_alpha=overlay_alpha,
        max_boxes=max_boxes,
    )


@app.post("/analyze-base64")
async def analyze_base64_image(request: Base64ImageRequest) -> Dict[str, Any]:
    """Analyze an image sent as base64 JSON."""
    b64 = request.image_base64.strip()

    # Accept both plain base64 and data URLs:
    # data:image/png;base64,AAAA...
    if "," in b64 and b64.lower().startswith("data:"):
        b64 = b64.split(",", 1)[1]

    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {exc}") from exc

    max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Maximum upload size is {MAX_UPLOAD_MB:g} MB.",
        )

    image = open_image_from_bytes(raw)

    return run_segmentation(
        image,
        filename=request.filename,
        max_side=request.max_side,
        min_area_px=request.min_area_px,
        return_overlay=request.return_overlay,
        overlay_alpha=request.overlay_alpha,
        max_boxes=request.max_boxes,
    )


# Local development only. Cloud Run uses the Docker CMD.
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
