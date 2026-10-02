import pandas as pd

from src.preprocess import preprocess_data


def test_preprocess_creates_derived_features(tmp_path, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "PROCESSED_DATA_PATH", str(tmp_path / "processed.csv"))
    frame = pd.DataFrame({
        "Last_Service_Date": ["2025-01-01", "2025-01-02"],
        "Warranty_Expiry_Date": ["2026-01-01", "2026-01-02"],
        "Tire_Condition": ["Good", "New"], "Brake_Condition": ["New", "Good"], "Battery_Status": ["Weak", "Good"],
        "Maintenance_History": ["Average", "Good"], "Vehicle_Model": ["Car", "SUV"], "Fuel_Type": ["Petrol", "Diesel"],
        "Transmission_Type": ["Manual", "Automatic"], "Owner_Type": ["First", "Second"], "Need_Maintenance": [1, 0],
    })
    out = preprocess_data(frame)
    assert out.loc[0, "condition_score"] == 1.0
    assert out.loc[0, "days_since_last_service"] == 181
    assert "Vehicle_Model_SUV" in out.columns
