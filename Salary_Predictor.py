import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
rf_model = joblib.load("saved files/rf_model_94.joblib")
le_gender = joblib.load("saved files/le_gender_encoder.joblib")
le_job = joblib.load("saved files/le_job_encoder.joblib")
expected_columns = joblib.load("saved files/expected_columns.joblib")

# Set page title
st.title("Employee Salary Predictor")
st.markdown("Provide employee details to estimate the expected salary.")

# Input form
experience = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)
age = st.number_input("Age", min_value=18, max_value=100, step=1)

gender = st.selectbox("Gender", le_gender.classes_)
job_title_options = sorted(le_job.classes_)
job_title = st.selectbox("Job Title", job_title_options)

education_level_options = sorted(set(
    col.replace("Education Level_", "") 
    for col in expected_columns 
    if col.startswith("Education Level_")
))
education_level = st.selectbox("Education Level", education_level_options)

# Predict button
if st.button("Predict Salary"):
    # Validation: Experience should not exceed (Age - 18)
    if experience > (age - 18):
        st.error("⚠️ Work Experience cannot exceed (Age - 18). Please correct the values.")
    else:
        # Encode inputs
        gender_encoded = le_gender.transform([gender])[0]
        job_title_encoded = le_job.transform([job_title])[0]

        input_df = pd.DataFrame({
            'Years of Experience': [experience],
            'Age': [age],
            'Gender': [gender_encoded],
            'Job Title': [job_title_encoded],
            'Education Level': [education_level]
        })

        # One-hot encoding for education level
        input_df = pd.get_dummies(input_df, columns=['Education Level'])

        # Add missing dummy columns (if any)
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]

        # Prediction
        salary = rf_model.predict(input_df)[0]

        # Display result
        st.success(f"💰 Predicted Salary: ₹{salary:,.2f}")




    
     

