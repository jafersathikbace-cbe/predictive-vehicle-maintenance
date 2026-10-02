import pandas as pd

from src.inference import prepare_features, predict_probability
from src.config import NUMERIC_FEATURES


class DummyScaler:
    def transform(self, values):
        return values


class DummyModel:
    def predict_proba(self, frame):
        return [[0.35, 0.65]]


def test_prepare_features_aligns_columns():
    input_data = {name: 100 for name in NUMERIC_FEATURES}
    frame = prepare_features(input_data, NUMERIC_FEATURES, DummyScaler())
    assert list(frame.columns) == NUMERIC_FEATURES


def test_predict_probability():
    frame = pd.DataFrame([[1]], columns=["Mileage"])
    assert predict_probability(DummyModel(), frame) == 0.65
