# Architecture

The project follows a small reproducible ML pipeline:

1. `src/data_loader.py` reads the raw CSV.
2. `src/preprocess.py` derives date and condition features and encodes categorical variables.
3. `run_eda.py` generates diagnostic visualizations.
4. `run_training.py` trains three classifiers and compares class-1 recall, precision, F1, and ROC-AUC.
5. `src/inference.py` loads the selected model/scaler and aligns application inputs with the training schema.
6. `src/explain.py` exposes SHAP explanations for tree models.
7. `app.py` provides the Streamlit UI.
