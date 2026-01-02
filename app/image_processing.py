# app/image_processing.py
from __future__ import annotations

import io
import os
from typing import Optional, Tuple

from PIL import Image

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency at runtime
    YOLO = None

from app.storage import download_media


_YOLO_MODEL = None


def _ensure_yolo_model_path() -> Optional[str]:
    model_path = os.getenv("YOLOV8_MODEL_PATH", "").strip()
    if model_path:
        return model_path if os.path.exists(model_path) else None

    bucket_name = os.getenv("YOLO_BUCKET_NAME", "").strip()
    model_name = os.getenv("YOLO_MODEL_NAME", "").strip()
    if not bucket_name or not model_name:
        return None

    local_dir = "/tmp/yolo"
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, model_name)

    if os.path.exists(local_path):
        return local_path

    try:
        data = download_media(bucket_name, model_name)
        with open(local_path, "wb") as f:
            f.write(data)
        return local_path
    except Exception as exc:
        print("WARN image_processing: failed to download YOLO model:", exc)
        return None


def _load_yolo_model() -> Optional["YOLO"]:
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    if YOLO is None:
        return None

    model_path = _ensure_yolo_model_path()
    if not model_path:
        return None

    _YOLO_MODEL = YOLO(model_path)
    return _YOLO_MODEL


def detect_single_bike_bbox(image: Image.Image) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
    """Return a single bicycle bbox (x1, y1, x2, y2) or a warning string."""
    model = _load_yolo_model()
    if model is None:
        return None, "Bike detection unavailable; saved original image."

    results = model(image, verbose=False)
    if not results:
        return None, "No bike detected; saved original image."

    result = results[0]
    if not hasattr(result, "boxes") or result.boxes is None:
        return None, "No bike detected; saved original image."

    names = getattr(model, "names", {})
    bikes = []
    for box in result.boxes:
        cls_id = int(box.cls[0]) if hasattr(box, "cls") else None
        label = names.get(cls_id, None) if cls_id is not None else None
        if label != "bicycle":
            continue
        conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
        xyxy = box.xyxy[0].tolist()
        bikes.append((conf, xyxy))

    if len(bikes) == 0:
        return None, "No bike detected; saved original image."
    if len(bikes) > 1:
        return None, "Multiple bikes detected; saved original image."

    _, xyxy = bikes[0]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    return (x1, y1, x2, y2), None


def crop_and_resize_webp(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    long_edge_px: Optional[int],
    quality: int = 85,
) -> bytes:
    """Crop to bbox and return a WebP (resized if long_edge_px set)."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, image.width))
    x2 = max(0, min(x2, image.width))
    y1 = max(0, min(y1, image.height))
    y2 = max(0, min(y2, image.height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop box from detector.")

    cropped = image.crop((x1, y1, x2, y2))

    if long_edge_px:
        long_edge = max(cropped.width, cropped.height)
        if long_edge > long_edge_px:
            scale = long_edge_px / long_edge
            new_w = max(1, int(round(cropped.width * scale)))
            new_h = max(1, int(round(cropped.height * scale)))
            cropped = cropped.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    cropped.save(buf, format="WEBP", quality=quality, method=6)
    return buf.getvalue()


def open_image_from_bytes(content: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(content))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
