import json
from io import BytesIO
from pathlib import Path

import pytest
import torch
from httpx import ASGITransport, AsyncClient
from PIL import Image

from api.main import app, reset_model_cache
from plant_disease_visionops.inference.predict_image import main as prediction_main
from plant_disease_visionops.inference.predictor import (
    DIAGNOSTIC_WARNING,
    load_model_for_inference,
    predict_image,
    preprocess_single_image,
)
from plant_disease_visionops.models.factory import create_model
from plant_disease_visionops.training.checkpointing import save_checkpoint


def _create_toy_inference_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    mapping = {"healthy": 0, "rust": 1, "scab": 2}
    (processed_dir / "class_to_index.json").write_text(
        json.dumps(mapping), encoding="utf-8"
    )

    model = create_model("baseline_cnn", num_classes=3, pretrained=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier[-1].bias.copy_(torch.tensor([3.0, 1.0, -1.0]))
    checkpoint_path = tmp_path / "models" / "best_model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        epoch=1,
        metrics={"macro_f1": 0.0},
        metadata={
            "model_name": "baseline_cnn",
            "model_config": {"num_classes": 3, "dropout": 0.3},
            "class_to_index": mapping,
        },
    )

    image_path = tmp_path / "leaf.png"
    Image.new("L", (24, 20), color=120).save(image_path)
    return processed_dir, checkpoint_path, image_path


def _image_upload() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (20, 20), color=(40, 130, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_single_image_preprocessing_converts_to_rgb_tensor(tmp_path: Path) -> None:
    image_path = tmp_path / "grayscale.png"
    Image.new("L", (21, 17), color=100).save(image_path)

    tensor = preprocess_single_image(image_path, image_size=16)

    assert tensor.shape == (3, 16, 16)
    assert tensor.dtype == torch.float32


def test_predict_image_returns_ranked_top_k_from_toy_checkpoint(tmp_path: Path) -> None:
    processed_dir, checkpoint_path, image_path = _create_toy_inference_files(tmp_path)
    inference_model = load_model_for_inference(checkpoint_path, processed_dir, device="cpu")

    result = predict_image(inference_model, image_path, image_size=16, top_k=2)

    assert result["predicted_label"] == "healthy"
    assert result["predicted_class_index"] == 0
    assert result["confidence"] == pytest.approx(result["top_k"][0]["probability"])
    assert [prediction["label"] for prediction in result["top_k"]] == ["healthy", "rust"]
    assert result["warning"] == DIAGNOSTIC_WARNING


def test_prediction_cli_smoke_test(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    processed_dir, checkpoint_path, image_path = _create_toy_inference_files(tmp_path)

    exit_code = prediction_main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--processed-dir",
            str(processed_dir),
            "--image-path",
            str(image_path),
            "--image-size",
            "16",
            "--top-k",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["predicted_label"] == "healthy"
    assert len(output["top_k"]) == 2
    assert "not a real agricultural diagnostic" in output["warning"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_fastapi_health_works_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PLANT_DISEASE_CHECKPOINT", raising=False)
    reset_model_cache()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checkpoint_configured": False,
        "checkpoint_available": False,
        "model_loaded": False,
    }


@pytest.mark.anyio
async def test_fastapi_predict_uses_toy_checkpoint_and_logs_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_dir, checkpoint_path, _ = _create_toy_inference_files(tmp_path)
    log_path = tmp_path / "logs" / "predictions.jsonl"
    monkeypatch.setenv("PLANT_DISEASE_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setenv("PLANT_DISEASE_PROCESSED_DIR", str(processed_dir))
    monkeypatch.setenv("PLANT_DISEASE_DEVICE", "cpu")
    monkeypatch.setenv("PLANT_DISEASE_IMAGE_SIZE", "16")
    monkeypatch.setenv("PLANT_DISEASE_LOG_PATH", str(log_path))
    reset_model_cache()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/predict?top_k=2",
            files={"file": ("leaf.png", _image_upload(), "image/png")},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["predicted_label"] == "healthy"
    assert result["predicted_class_index"] == 0
    assert len(result["top_k"]) == 2
    log_record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert log_record["filename"] == "leaf.png"
    assert log_record["predicted_label"] == "healthy"
    reset_model_cache()


@pytest.mark.anyio
async def test_fastapi_predict_fails_clearly_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PLANT_DISEASE_CHECKPOINT", raising=False)
    reset_model_cache()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/predict",
            files={"file": ("leaf.png", _image_upload(), "image/png")},
        )

    assert response.status_code == 503
    assert "PLANT_DISEASE_CHECKPOINT" in response.json()["detail"]
