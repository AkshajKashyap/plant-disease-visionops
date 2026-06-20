"""Shared single-image inference utilities for local interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from torch import Tensor, nn

from plant_disease_visionops.data.dataset import load_class_mapping
from plant_disease_visionops.data.transforms import build_eval_transform
from plant_disease_visionops.evaluation.model_loading import load_trained_model
from plant_disease_visionops.training.reproducibility import get_device

DIAGNOSTIC_WARNING = (
    "This prediction is for demonstration only and is not a real agricultural diagnostic."
)


class ImagePreprocessingError(RuntimeError):
    """Raised when an inference image cannot be decoded or transformed."""


class TopPrediction(TypedDict):
    """One ranked model prediction."""

    label: str
    class_index: int
    probability: float


class PredictionResult(TypedDict):
    """Serializable result returned by every inference interface."""

    predicted_label: str
    predicted_class_index: int
    confidence: float
    top_k: list[TopPrediction]
    model_name: str
    device: str
    warning: str


@dataclass(frozen=True, slots=True)
class InferenceModel:
    """A loaded classifier and the metadata required to decode its logits."""

    model: nn.Module
    model_name: str
    class_to_index: dict[str, int]
    index_to_class: tuple[str, ...]
    device: torch.device


ImageSource = Path | str | Image.Image


def _ordered_classes(class_to_index: dict[str, int]) -> tuple[str, ...]:
    return tuple(sorted(class_to_index, key=class_to_index.__getitem__))


def load_model_for_inference(
    checkpoint_path: Path | str,
    processed_dir: Path | str,
    device: str | torch.device = "auto",
) -> InferenceModel:
    """Load a project checkpoint and verify its processed class mapping."""
    resolved_device = get_device(str(device))
    loaded = load_trained_model(checkpoint_path, device=resolved_device)
    processed_path = Path(processed_dir).expanduser()
    processed_mapping = load_class_mapping(processed_path / "class_to_index.json")
    if loaded.class_to_index != processed_mapping:
        raise ValueError("Checkpoint class mapping does not match processed class_to_index.json")
    return InferenceModel(
        model=loaded.model,
        model_name=loaded.model_name,
        class_to_index=loaded.class_to_index,
        index_to_class=_ordered_classes(loaded.class_to_index),
        device=resolved_device,
    )


def _load_rgb_image(source: ImageSource) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    image_path = Path(source).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except OSError as exc:
        raise ImagePreprocessingError(f"Could not decode image {image_path}: {exc}") from exc


def preprocess_single_image(source: ImageSource, image_size: int = 128) -> Tensor:
    """Convert one path or Pillow image into a normalized CHW tensor."""
    if image_size <= 0:
        raise ValueError(f"image_size must be greater than zero; got {image_size}")
    rgb_image = _load_rgb_image(source)
    try:
        tensor = build_eval_transform(image_size)(rgb_image)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ImagePreprocessingError(f"Could not preprocess image: {exc}") from exc
    if not isinstance(tensor, Tensor):
        raise ImagePreprocessingError(
            f"Preprocessing returned {type(tensor).__name__} instead of a tensor"
        )
    if tensor.shape != (3, image_size, image_size):
        raise ImagePreprocessingError(
            f"Preprocessing returned an unexpected tensor shape: {tuple(tensor.shape)}"
        )
    return tensor


def predict_image(
    inference_model: InferenceModel,
    source: ImageSource,
    image_size: int = 128,
    top_k: int = 5,
) -> PredictionResult:
    """Run one model prediction and return ranked labels and probabilities."""
    if top_k <= 0:
        raise ValueError(f"top_k must be greater than zero; got {top_k}")
    actual_top_k = min(top_k, len(inference_model.index_to_class))
    image_tensor = preprocess_single_image(source, image_size).unsqueeze(0)

    inference_model.model.eval()
    with torch.inference_mode():
        logits = inference_model.model(image_tensor.to(inference_model.device))
        if logits.shape != (1, len(inference_model.index_to_class)):
            raise RuntimeError(
                "Model returned unexpected logits shape: "
                f"{tuple(logits.shape)}; expected (1, {len(inference_model.index_to_class)})"
            )
        probabilities = torch.softmax(logits, dim=1)[0]
        top_probabilities, top_indices = probabilities.topk(actual_top_k)

    ranked: list[TopPrediction] = []
    for probability, class_index in zip(
        top_probabilities.cpu().tolist(),
        top_indices.cpu().tolist(),
        strict=True,
    ):
        ranked.append(
            {
                "label": inference_model.index_to_class[class_index],
                "class_index": class_index,
                "probability": float(probability),
            }
        )

    winner = ranked[0]
    return {
        "predicted_label": winner["label"],
        "predicted_class_index": winner["class_index"],
        "confidence": winner["probability"],
        "top_k": ranked,
        "model_name": inference_model.model_name,
        "device": str(inference_model.device),
        "warning": DIAGNOSTIC_WARNING,
    }
