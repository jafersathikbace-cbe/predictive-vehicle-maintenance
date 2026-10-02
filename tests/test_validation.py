import pytest

from src.validation import maintenance_label, validate_probability


def test_probability_validation():
    assert validate_probability(0.25) == 0.25
    with pytest.raises(ValueError):
        validate_probability(1.2)


def test_maintenance_label():
    assert maintenance_label(0.8) == "Maintenance Needed"
    assert maintenance_label(0.2) == "No Immediate Need"
