"""Input validation and prediction display helpers."""


def validate_probability(probability: float) -> float:
    """Ensure a model probability is finite and within the expected range."""
    value = float(probability)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Prediction probability must be between 0 and 1")
    return value


def maintenance_label(probability: float, threshold: float = 0.5) -> str:
    """Convert a maintenance probability into the UI label."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1")
    return "Maintenance Needed" if validate_probability(probability) >= threshold else "No Immediate Need"
