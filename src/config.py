# src/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "vehicle_maintenance_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "vehicle_maintenance_processed.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.pkl"
LEGACY_MODEL_PATH = MODEL_DIR / "xgboost_best.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
REFERENCE_DATE = "2025-07-01"  # You can change this to today's date if you want

NUMERIC_FEATURES = [
    'Mileage', 'Reported_Issues', 'Vehicle_Age', 'Engine_Size',
    'Odometer_Reading', 'Insurance_Premium', 'Service_History',
    'Accident_History', 'Fuel_Efficiency', 'days_since_last_service',
    'days_to_warranty_end', 'condition_score'
]