"""Randomness and device selection helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and configure deterministic CUDA behavior."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto") -> torch.device:
    """Resolve an available training device or fail clearly for unavailable requests."""
    choice = requested.lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if choice == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available")
    if choice not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"device must be auto, cpu, cuda, or mps; got {requested!r}")
    return torch.device(choice)
