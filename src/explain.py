"""SHAP explanation helpers for the maintenance classifier."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


def build_explainer(model: Any) -> Any:
    """Build a SHAP TreeExplainer for the trained tree model."""
    return shap.TreeExplainer(model)


def explain_prediction(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return SHAP values for the supplied feature row."""
    values = build_explainer(model)(features)
    return np.asarray(values.values if hasattr(values, 'values') else values)
