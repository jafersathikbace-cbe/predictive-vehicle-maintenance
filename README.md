# Predictive Vehicle Maintenance

A machine-learning application that predicts whether a vehicle is likely to require maintenance and presents an interpretable result through a Streamlit interface.

## Highlights

- 50,000-row vehicle maintenance dataset
- Date-derived service and warranty features
- Ordinal and one-hot feature engineering
- XGBoost, LightGBM, and Random Forest comparison
- Stratified train/test evaluation with ROC-AUC, precision, recall, and F1
- Feature scaling for numeric variables
- SHAP-based model explanation support
- Streamlit prediction interface

## Project structure

```text
data/      raw and processed datasets
models/     trained model and scaler artifacts
src/        reusable loading, preprocessing, inference and explanation code
tests/      automated tests
run_eda.py
run_preprocess.py
run_training.py
app.py
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the pipeline

```bash
python run_preprocess.py
python run_eda.py
python run_training.py
streamlit run app.py
```

Run tests with `pytest -q`.

## Responsible use

This project is a predictive demonstration, not a substitute for professional vehicle inspection or maintenance advice. Model performance depends on the training data and should be monitored when the data distribution changes.

## Notes

The included model artifact allows the Streamlit app to run without retraining. The reference date used for time-based features is configured in `src/config.py`. For production use, retrain and validate the model against current operational data before deployment.
