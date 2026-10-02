from pathlib import Path

from src.config import MODEL_PATH, SCALER_PATH, RAW_DATA_PATH


def test_required_artifacts_exist():
    assert Path(RAW_DATA_PATH).exists()
    assert Path(MODEL_PATH).exists()
    assert Path(SCALER_PATH).exists()
