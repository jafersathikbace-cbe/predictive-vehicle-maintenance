# run_training.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from src.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE
)

def train_and_evaluate():
    print("=== Phase 5–6: Modeling, Training & Evaluation ===")
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Features & target
    X = df.drop(columns=['Need_Maintenance'])
    y = df['Need_Maintenance']

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # Scale only numeric features
    scaler = StandardScaler()
    X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_test[NUMERIC_FEATURES]  = scaler.transform(X_test[NUMERIC_FEATURES])

    # ────────────────────────────────────────────────
    # Models (with imbalance handling)
    # ────────────────────────────────────────────────
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.07,
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum(),  # ≈ 0.235
            random_state=RANDOM_STATE,
            eval_metric='aucpr',
            n_jobs=-1,
            verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.08,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "AUC": auc,
            "Recall_1": report['1']['recall'],
            "Precision_1": report['1']['precision'],
            "F1_1": report['1']['f1-score']
        }

        print(f"\n{name} Results:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm)
