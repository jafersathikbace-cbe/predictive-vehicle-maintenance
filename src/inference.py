"""Model loading and prediction helpers shared by the application."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_PATH, SCALER_PATH, NUMERIC_FEATURES


def load_artifacts() -> tuple[Any, Any]:
    """Load the trained model and feature scaler."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler
