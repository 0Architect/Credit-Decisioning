import streamlit as st
import requests

# API URL
API_URL = "http://localhost:5000/predict"  # Change to your ngrok URL when deploying

# UI Elements
st.title("Credit Score Prediction")

st.write("Enter the details of the borrower:")

# Create form fields
revolving_utilization = st.number_input("Revolving Utilization of Unsecured Lines (%)", min_value=0.0, max_value=1.0)
age = st.number_input("Age", min_value=18, max_value=100)
debt_ratio = st.number_input("Debt Ratio (%)", min_value=0.0, max_value=1.0)
monthly_income = st.number_input("Monthly Income ($)", min_value=0)
open_credit_lines = st.number_input("Number of Open Credit Lines and Loans", min_value=0)
time_30_59_days = st.number_input("Number of times 30-59 days late in last 2 years", min_value=0)
time_60_89_days = st.number_input("Number of times 60-89 days late in last 2 years", min_value=0)
time_90_days = st.number_input("Number of times 90+ days late in last 2 years", min_value=0)
real_estate_loans = st.number_input("Number of Real Estate Loans or Lines", min_value=0)
dependents = st.number_input("Number of Dependents", min_value=0)

# Create a button to trigger the prediction
if st.button('Predict Credit Score'):
    # Gather data in dictionary format
    input_data = {
        "RevolvingUtilizationOfUnsecuredLines": revolving_utilization,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": time_30_59_days,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": time_90_days,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": time_60_89_days,
        "NumberOfDependents": dependents
    }

    # Send the data to the Flask API for prediction
    response = requests.post(API_URL, json=input_data)

    if response.status_code == 200:
        # Parse and display the result
        result = response.json()
        st.success(f"{'The borrower would not default' if result['credit_score'] == 0 else 'The borrower is likely to default on the loan.'}")
    else:
        st.error("Error: " + response.json().get('error', 'Something went wrong'))