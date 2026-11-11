"""Streamlit interface for the trained California housing model."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction")
st.write("Enter California Housing dataset features to get an educational model estimate.")

if not MODEL_PATH.exists():
    st.error("Model not found. First run: python train_model.py")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

st.info("This demo uses a California census-derived dataset. It is not a current India/Chennai property valuation tool.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    values = {}
    labels = {
        "MedInc": ("Median income (10,000 USD units)", 3.5, 0.5, 15.0),
        "HouseAge": ("Median house age (years)", 20.0, 1.0, 52.0),
        "AveRooms": ("Average rooms", 5.0, 1.0, 20.0),
        "AveBedrms": ("Average bedrooms", 1.0, 0.2, 5.0),
        "Population": ("Area population", 1000.0, 3.0, 35000.0),
        "AveOccup": ("Average household occupancy", 3.0, 1.0, 20.0),
        "Latitude": ("Latitude", 34.0, 32.0, 42.0),
        "Longitude": ("Longitude", -119.0, -125.0, -114.0),
    }
    for i, feature in enumerate(features):
        label, default, minimum, maximum = labels[feature]
        container = col1 if i % 2 == 0 else col2
        values[feature] = container.number_input(
            label, min_value=float(minimum), max_value=float(maximum),
            value=float(default), step=0.1
        )
    submitted = st.form_submit_button("Predict estimated value")

if submitted:
    row = pd.DataFrame([values], columns=features)
    prediction_100k = float(model.predict(row)[0])
    # California dataset target is expressed in $100,000 units.
    estimated_usd = max(0, prediction_100k * 100000)
    st.metric("Model estimate (USD)", f"${estimated_usd:,.0f}")
    st.caption(f"Model: {bundle['best_model_name']}. This is a dataset-scale estimate, not a real offer or appraisal.")
