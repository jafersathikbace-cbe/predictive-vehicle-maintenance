# src/preprocess.py
import pandas as pd
from src import config

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting preprocessing...")

    # Convert dates
    df['Last_Service_Date'] = pd.to_datetime(df['Last_Service_Date'])
    df['Warranty_Expiry_Date'] = pd.to_datetime(df['Warranty_Expiry_Date'])

    # Time features
