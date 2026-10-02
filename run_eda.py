# run_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PROCESSED_DATA_PATH

# Better looking plots
plt.style.use('seaborn-v0_8-whitegrid')

def run_eda():
    print("=== Phase 4: Exploratory Data Analysis ===")
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print("Shape:", df.shape, "\n")

    target = 'Need_Maintenance'

    # ──────────────── 1. Target distribution ────────────────
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target, data=df, palette='viridis')
    plt.title("Target Distribution\n(Need_Maintenance: 1 = Yes, 0 = No)")
    plt.xlabel("Need Maintenance")
    plt.ylabel("Count")
    plt.savefig("eda_target_distribution.png", dpi=120, bbox_inches='tight')
    plt.close()
