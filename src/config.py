# src/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "vehicle_maintenance_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "vehicle_maintenance_processed.csv"

