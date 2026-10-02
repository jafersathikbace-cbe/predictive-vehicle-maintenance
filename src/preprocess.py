# src/preprocess.py
import pandas as pd
from src import config

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting preprocessing...")

    # Convert dates
    df['Last_Service_Date'] = pd.to_datetime(df['Last_Service_Date'])
    df['Warranty_Expiry_Date'] = pd.to_datetime(df['Warranty_Expiry_Date'])

    # Time features
    ref_date = pd.to_datetime(config.REFERENCE_DATE)
    df['days_since_last_service'] = (ref_date - df['Last_Service_Date']).dt.days
    df['days_to_warranty_end'] = (df['Warranty_Expiry_Date'] - ref_date).dt.days

    # Ordinal encoding
    condition_map = {'New': 2, 'Good': 1, 'Worn Out': 0, 'Weak': 0}
    history_map = {'Good': 2, 'Average': 1, 'Poor': 0}

    df['Tire_num'] = df['Tire_Condition'].map(condition_map)
    df['Brake_num'] = df['Brake_Condition'].map(condition_map)
    df['Battery_num'] = df['Battery_Status'].map(condition_map)
    df['History_num'] = df['Maintenance_History'].map(history_map)

    df['condition_score'] = (df['Tire_num'] + df['Brake_num'] + df['Battery_num']) / 3

    # One-hot encoding
    onehot_cols = ['Vehicle_Model', 'Fuel_Type', 'Transmission_Type', 'Owner_Type']
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True, dtype=int)

    # Drop original string/date columns
    drop_cols = [
        'Maintenance_History', 'Tire_Condition', 'Brake_Condition', 'Battery_Status',
        'Last_Service_Date', 'Warranty_Expiry_Date'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print("Preprocessing complete. Final shape:", df.shape)
    print("Columns:", df.columns.tolist())

    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"Processed file saved → {config.PROCESSED_DATA_PATH}")

    return df