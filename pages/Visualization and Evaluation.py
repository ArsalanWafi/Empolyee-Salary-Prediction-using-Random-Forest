import streamlit as st
import pandas as pd
import numpy as np
import joblib
# from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page config
st.set_page_config(page_title="Visualization Dashboard", layout="wide")

st.title("Salary Model Visualization Dashboard")

# Outlier Visualization
st.header("Outlier Detection Comparison")

# Salary vs Age
st.subheader("Salary vs Age")
col1, col2 = st.columns(2)
with col1:
    st.image("images/salary_by_age_before_outlier_removal.png", caption="Before Outlier Removal", use_container_width=True)
with col2:
    st.image("images/salary_by_age_after_outlier_removal.png", caption="After Outlier Removal", use_container_width=True)

# Salary vs Experience
st.subheader("Salary vs Years of Experience")
col3, col4 = st.columns(2)
with col3:
    st.image("images/salary_by_Experience_before_outlier_removal.png", caption="Before Outlier Removal", use_container_width=True)
with col4:
    st.image("images/salary_by_Experience_before_outlier_removal.png", caption="After Outlier Removal", use_container_width=True)

# Feature Importance
st.header("Feature Importance (Random Forest)")
st.image("images/Feature_Importance.png", caption="Random Forest Feature Importances", use_container_width=True)

# Actual vs Predicted
st.header("Actual vs Predicted Salary")
st.image("images/actual_vs_predicted.png", caption="Actual vs Predicted Plot", use_container_width=True)

# Model Evaluation Metrics
st.header("Model Evaluation Metrics")
metrics = joblib.load("saved files/model_metrics.joblib")

# Load saved data
# rf_model = joblib.load("saved files/rf_model_94.joblib")
# le_gender = joblib.load("saved files/le_gender_encoder.joblib")
# le_job = joblib.load("saved files/le_job_encoder.joblib")
# expected_columns = joblib.load("saved files/expected_columns.joblib")


st.header("Model Evaluation Metrics")
st.metric("R² Score", f"{metrics['R2_Score']:.4f}")
st.metric("Accuracy", f"{metrics['Accuracy (%)']:.2f}%")
st.metric("Mean Squared Error (MSE)", f"{metrics['MSE']:.2f}")
st.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE (%)']:.2f}%")
