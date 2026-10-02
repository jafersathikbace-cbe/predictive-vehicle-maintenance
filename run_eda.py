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

    print("Target distribution (%):")
    print(df[target].value_counts(normalize=True).mul(100).round(2))
    print()

    # ──────────────── 2. Top correlations with target ────────────────
    corr_with_target = df.corr(numeric_only=True)[target].abs().sort_values(ascending=False)
    top_corr = corr_with_target.head(15)

    print("Top 15 features correlated with Need_Maintenance (absolute value):")
    print(top_corr.round(3))
    print()

    # Bar plot of top correlations
    plt.figure(figsize=(10, 6))
    top_corr.plot(kind='bar', color='teal')
    plt.title("Top Features by Correlation with Need_Maintenance")
    plt.ylabel("Absolute Correlation")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("eda_top_correlations.png", dpi=120, bbox_inches='tight')
    plt.close()

    # ──────────────── 3. Boxplots of most promising numerical features ────────────────
    key_features = [
        'days_since_last_service',
        'condition_score',
        'Mileage',
        'Reported_Issues',
        'Vehicle_Age',
        'days_to_warranty_end'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
