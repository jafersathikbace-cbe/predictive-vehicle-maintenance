# app.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from src.config import NUMERIC_FEATURES
from src.inference import load_artifacts, prepare_features, predict_probability
from src.validation import maintenance_label, validate_probability

# ─── Clean, professional, minimal styling ───
st.markdown("""
    <style>
    .main {background-color: #f9fafb; padding: 2rem;}
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }
    .stButton>button:hover {background-color: #1d4ed8;}
    .stAlert {border-radius: 8px;}
    h1, h2, h3 {color: #1f2937;}
    .report-box {
        background-color: #f1f5f9;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Vehicle Maintenance Predictor", layout="wide")

st.title("Vehicle Maintenance Predictor")
st.markdown("Enter vehicle details to get instant prediction, risk level and explanation.")

# ─── Load model & scaler ───
@st.cache_resource
def load_model_and_scaler():
    return load_artifacts()

model, scaler = load_model_and_scaler()

# ─── Load exact feature names ───
from src.config import DATA_DIR

feature_names = pd.read_csv(DATA_DIR / "feature_names.csv", header=None)[0].tolist()

# ─── Full Input Form ───
with st.form("vehicle_form"):
    st.subheader("Vehicle Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        mileage = st.number_input("Mileage (km)", 30000, 80000, 55000, step=1000)
        vehicle_age = st.slider("Vehicle Age (years)", 1, 10, 5)
        reported_issues = st.slider("Reported Issues", 0, 5, 2)
        days_since_service = st.number_input("Days since last service", 0, 1000, 180)

    with col2:
        vehicle_model = st.selectbox("Vehicle Model", ["Bus", "Car", "Motorcycle", "SUV", "Truck", "Van"])
        fuel_type = st.selectbox("Fuel Type", ["Diesel", "Electric", "Petrol"])
        transmission = st.selectbox("Transmission Type", ["Automatic", "Manual"])
        owner_type = st.selectbox("Owner Type", ["First", "Second", "Third"])

    with col3:
        tire_cond = st.selectbox("Tire Condition", ["New", "Good", "Worn Out"], index=1)
        brake_cond = st.selectbox("Brake Condition", ["New", "Good", "Worn Out"], index=1)
        battery_stat = st.selectbox("Battery Status", ["New", "Good", "Weak"], index=1)
        fuel_eff = st.slider("Fuel Efficiency (km/l)", 10.0, 20.0, 15.0)

    submit = st.form_submit_button("Get Prediction", type="primary", use_container_width=True)

if submit:
    with st.spinner("Analyzing your vehicle..."):
        # ─── Prepare input ───
        tire_score    = {"New": 2, "Good": 1, "Worn Out": 0}[tire_cond]
        brake_score   = {"New": 2, "Good": 1, "Worn Out": 0}[brake_cond]
        battery_score = {"New": 2, "Good": 1, "Weak": 0}[battery_stat]

        input_data = {
            'Mileage': float(mileage),
            'Reported_Issues': reported_issues,
            'Vehicle_Age': vehicle_age,
            'Fuel_Efficiency': fuel_eff,
            'days_since_last_service': days_since_service,
            'Tire_num': tire_score,
            'Brake_num': brake_score,
            'Battery_num': battery_score,
            'condition_score': (tire_score + brake_score + battery_score) / 3.0,
            'Engine_Size': 1500.0,
            'Odometer_Reading': mileage + 50000,
            'Insurance_Premium': 15000.0,
            'Service_History': 5,
            'Accident_History': 1,
            'days_to_warranty_end': 365,
            'History_num': 1,
        }

        # One-hot encoding
        for vm in ["Car", "Motorcycle", "SUV", "Truck", "Van"]:
            input_data[f'Vehicle_Model_{vm}'] = 1 if vehicle_model == vm else 0
        for ft in ["Electric", "Petrol"]:
            input_data[f'Fuel_Type_{ft}'] = 1 if fuel_type == ft else 0
        input_data['Transmission_Type_Manual'] = 1 if transmission == "Manual" else 0
        for ot in ["Second", "Third"]:
            input_data[f'Owner_Type_{ot}'] = 1 if owner_type == ot else 0

        input_df = prepare_features(input_data, feature_names, scaler)

        # ─── Prediction ───
        proba = predict_probability(model, input_df)
        proba = validate_probability(proba)
        prediction_text = maintenance_label(proba)

        # ─── Show result ───
        st.markdown("### Prediction Result")
        if proba >= 0.5:
            st.error(f"**{prediction_text}**  \n**Probability: {proba:.1%}**")
        else:
            st.success(f"**{prediction_text}**  \n**Probability: {proba:.1%}**")

        # Progress bar (fixed type)
        st.progress(float(proba))
        st.caption(f"Risk Level: {proba:.1%}")

        # ─── SHAP + Report ───
        st.markdown("### Risk Analysis & Explanation")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)[0]

        # Top factors
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values
        }).sort_values('SHAP Value', ascending=False).head(8)

        st.write("**Top factors influencing prediction** (positive = increases maintenance risk):")
        for _, row in shap_df.iterrows():
            direction = "↑ increases risk" if row['SHAP Value'] > 0 else "↓ decreases risk"
            st.markdown(f"- **{row['Feature']}**: {row['SHAP Value']:.3f} {direction}")

        summary = (
            "High risk — likely due to overdue service, poor component condition, and/or high mileage."
            if proba >= 0.5 else
            "Low risk — vehicle appears well-maintained with recent service and good overall condition."
        )
        st.info(summary)

        # SHAP bar plot (stable & always visible)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#ef4444' if x > 0 else '#10b981' for x in shap_df['SHAP Value']]
        shap_df.plot.barh(x='Feature', y='SHAP Value', ax=ax, color=colors)
        ax.set_xlabel("SHAP Value (impact on prediction)")
        ax.set_title("Top Features by Impact")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.caption("Demo using synthetic data – for educational purposes only.")