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


def prepare_features(input_data: dict, feature_names: list[str], scaler: Any) -> pd.DataFrame:
    """Align application inputs with the training feature schema and scale numerics."""
    frame = pd.DataFrame([input_data]).reindex(columns=feature_names, fill_value=0.0)
    frame[NUMERIC_FEATURES] = scaler.transform(frame[NUMERIC_FEATURES])
    return frame


def predict_probability(model: Any, features: pd.DataFrame) -> float:
    """Return the positive-class maintenance probability."""
    return float(model.predict_proba(features)[0][1])
