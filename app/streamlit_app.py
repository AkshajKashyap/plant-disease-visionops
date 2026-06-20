"""Streamlit upload demo for a locally generated project checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from PIL import Image

from plant_disease_visionops.inference.predictor import (
    DIAGNOSTIC_WARNING,
    InferenceModel,
    load_model_for_inference,
    predict_image,
)


@st.cache_resource(show_spinner="Loading checkpoint...")
def _load_cached_model(checkpoint: str, processed_dir: str, device: str) -> InferenceModel:
    return load_model_for_inference(checkpoint, processed_dir, device)


def main() -> None:
    """Render the local image-upload inference demo."""
    st.set_page_config(page_title="Plant Disease VisionOps", layout="centered")
    st.title("Plant Disease VisionOps")
    st.caption("Local single-image checkpoint demo")
    st.warning(DIAGNOSTIC_WARNING)

    with st.sidebar:
        st.header("Model settings")
        checkpoint = st.text_input(
            "Checkpoint path",
            value=os.getenv("PLANT_DISEASE_CHECKPOINT", ""),
            placeholder="artifacts/models/.../best_model.pt",
        )
        processed_dir = st.text_input(
            "Processed metadata directory",
            value=os.getenv("PLANT_DISEASE_PROCESSED_DIR", "data/processed"),
        )
        device = st.selectbox("Device", options=("auto", "cpu", "cuda", "mps"), index=0)
        image_size = st.number_input("Image size", min_value=8, value=128, step=8)
        top_k = st.number_input("Top predictions", min_value=1, max_value=38, value=5, step=1)

    uploaded_file = st.file_uploader("Upload a leaf image", type=("jpg", "jpeg", "png"))
    if uploaded_file is None:
        st.info("Upload a JPEG or PNG image to begin.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except OSError as exc:
        st.error(f"Could not decode the uploaded image: {exc}")
        return
    st.image(image, caption=uploaded_file.name, use_container_width=True)

    if not checkpoint.strip():
        st.error(
            "Provide a checkpoint path in the sidebar or set PLANT_DISEASE_CHECKPOINT. "
            "Checkpoints are not included in this repository."
        )
        return
    if not Path(checkpoint).expanduser().is_file():
        st.error(f"Checkpoint does not exist: {checkpoint}")
        return

    if st.button("Classify image", type="primary"):
        try:
            inference_model = _load_cached_model(checkpoint, processed_dir, device)
            result = predict_image(
                inference_model,
                image,
                image_size=int(image_size),
                top_k=int(top_k),
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.subheader(result["predicted_label"])
        st.metric("Confidence", f"{result['confidence']:.2%}")
        st.table(
            [
                {
                    "Rank": rank,
                    "Label": prediction["label"],
                    "Class index": prediction["class_index"],
                    "Probability": f"{prediction['probability']:.2%}",
                }
                for rank, prediction in enumerate(result["top_k"], start=1)
            ]
        )
        st.warning(result["warning"])


if __name__ == "__main__":
    main()
