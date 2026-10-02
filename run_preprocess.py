# run_preprocess.py
from src.data_loader import load_raw_data
from src.preprocess import preprocess_data
import pandas as pd

if __name__ == "__main__":
    print("=== Starting Data Preprocessing Phase ===")
    df_raw = load_raw_data()
    df_processed = preprocess_data(df_raw)
    
    # Save exact feature names the model was trained on
    feature_names = df_processed.drop(columns=['Need_Maintenance']).columns.tolist()
    pd.Series(feature_names).to_csv(DATA_DIR / "feature_names.csv", index=False, header=False)
    print("Feature names saved → data/feature_names.csv")
    
    print("=== Preprocessing Finished Successfully ===")