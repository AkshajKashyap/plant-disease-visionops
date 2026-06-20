"""Lightweight, lazy-loading FastAPI application for local image inference."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from plant_disease_visionops.data.dataset import DatasetLoadingError
from plant_disease_visionops.evaluation.model_loading import ModelLoadingError
from plant_disease_visionops.inference.predictor import (
    ImagePreprocessingError,
    InferenceModel,
    PredictionResult,
    load_model_for_inference,
    predict_image,
)

LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CHECKPOINT_ENV = "PLANT_DISEASE_CHECKPOINT"
PROCESSED_DIR_ENV = "PLANT_DISEASE_PROCESSED_DIR"
DEVICE_ENV = "PLANT_DISEASE_DEVICE"
IMAGE_SIZE_ENV = "PLANT_DISEASE_IMAGE_SIZE"
TOP_K_ENV = "PLANT_DISEASE_TOP_K"
LOG_PATH_ENV = "PLANT_DISEASE_LOG_PATH"


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Runtime configuration read from environment variables."""

    checkpoint_path: Path | None
    processed_dir: Path
    device: str
    image_size: int
    top_k: int
    log_path: Path


_model_lock = threading.Lock()
_log_lock = threading.Lock()
_cached_model: InferenceModel | None = None
_cached_key: tuple[str, str, str] | None = None


def _positive_environment_integer(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value}")
    return value


def get_settings() -> ApiSettings:
    """Read API settings without hardcoded machine-specific paths."""
    checkpoint_value = os.getenv(CHECKPOINT_ENV)
    return ApiSettings(
        checkpoint_path=Path(checkpoint_value).expanduser() if checkpoint_value else None,
        processed_dir=Path(os.getenv(PROCESSED_DIR_ENV, "data/processed")).expanduser(),
        device=os.getenv(DEVICE_ENV, "auto"),
        image_size=_positive_environment_integer(IMAGE_SIZE_ENV, 128),
        top_k=_positive_environment_integer(TOP_K_ENV, 5),
        log_path=Path(
            os.getenv(LOG_PATH_ENV, "artifacts/logs/predictions.jsonl")
        ).expanduser(),
    )


def reset_model_cache() -> None:
    """Clear lazy model state, primarily for configuration changes and tests."""
    global _cached_key, _cached_model
    with _model_lock:
        _cached_model = None
        _cached_key = None


def _get_model(settings: ApiSettings) -> InferenceModel:
    global _cached_key, _cached_model
    if settings.checkpoint_path is None:
        raise FileNotFoundError(
            f"Checkpoint is not configured. Set the {CHECKPOINT_ENV} environment variable."
        )
    key = (
        str(settings.checkpoint_path.resolve()),
        str(settings.processed_dir.resolve()),
        settings.device,
    )
    with _model_lock:
        if _cached_model is None or _cached_key != key:
            _cached_model = load_model_for_inference(
                settings.checkpoint_path,
                settings.processed_dir,
                settings.device,
            )
            _cached_key = key
        return _cached_model


def _write_prediction_log(
    settings: ApiSettings,
    filename: str | None,
    result: PredictionResult,
) -> None:
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "filename": filename,
        "checkpoint": settings.checkpoint_path.name if settings.checkpoint_path else None,
        "predicted_label": result["predicted_label"],
        "predicted_class_index": result["predicted_class_index"],
        "confidence": result["confidence"],
        "top_k": result["top_k"],
    }
    try:
        with _log_lock:
            settings.log_path.parent.mkdir(parents=True, exist_ok=True)
            with settings.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        LOGGER.warning("Could not write prediction log %s: %s", settings.log_path, exc)


app = FastAPI(
    title="Plant Disease VisionOps Local Inference",
    version="0.1.0",
    description="Local demonstration API. Predictions are not agricultural diagnoses.",
)


@app.get("/health")
async def health() -> dict[str, object]:
    """Report service reachability and whether a checkpoint path is configured."""
    checkpoint_value = os.getenv(CHECKPOINT_ENV)
    return {
        "status": "ok",
        "checkpoint_configured": bool(checkpoint_value),
        "checkpoint_available": bool(
            checkpoint_value and Path(checkpoint_value).expanduser().is_file()
        ),
        "model_loaded": _cached_model is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int | None = Query(default=None, ge=1, le=100),
) -> PredictionResult:
    """Classify one uploaded JPEG or PNG image."""
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image exceeds the 10 MB limit")

    try:
        settings = get_settings()
        inference_model = _get_model(settings)
    except (
        DatasetLoadingError,
        ModelLoadingError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        with Image.open(BytesIO(contents)) as image:
            result = predict_image(
                inference_model,
                image,
                image_size=settings.image_size,
                top_k=top_k or settings.top_k,
            )
    except (ImagePreprocessingError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    _write_prediction_log(settings, file.filename, result)
    return result
