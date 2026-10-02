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

