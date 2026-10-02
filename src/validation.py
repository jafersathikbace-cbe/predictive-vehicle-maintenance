"""Input validation and prediction display helpers."""


def validate_probability(probability: float) -> float:
    """Ensure a model probability is finite and within the expected range."""
    value = float(probability)
    if not 0.0 <= value <= 1.0:
