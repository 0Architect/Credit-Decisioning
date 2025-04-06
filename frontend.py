import streamlit as st
import streamlit.components.v1 as components
import requests
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


API_URL = "https://credit-decisioning.onrender.com"

st.title("Credit Score Prediction")

st.write("Enter the details of the borrower:")

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

if st.button('Predict Credit Score'):
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

    response = requests.post(API_URL, json=input_data)

    if response.status_code == 200:
        result = response.json()
        st.success(f"{'The borrower would not default' if result['credit_score'] == 0 else 'The borrower is likely to default on the loan.'}")

        X = pd.DataFrame([result['features']])
        shap_values = np.array(result['shap_values'])
        expected_value = result['expected_value']

        desired_column_order = [
        'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
        ]
        X = X[desired_column_order]
        column_mapping = {
            'RevolvingUtilizationOfUnsecuredLines': 'Utilization',
            'age': 'Age',
            'NumberOfTime30-59DaysPastDueNotWorse': '30_59_Late',
            'DebtRatio': 'DebtRatio',  # No change for DebtRatio
            'MonthlyIncome': 'Income',
            'NumberOfOpenCreditLinesAndLoans': 'OpenLines',
            'NumberOfTimes90DaysLate': '90+_Late',
            'NumberRealEstateLoansOrLines': 'RealEstate',
            'NumberOfTime60-89DaysPastDueNotWorse': '60_89_Late',
            'NumberOfDependents': 'Dependents'
        }
        X.columns = [column_mapping[col] for col in desired_column_order]
        top_k = 7
        important_indices = np.argsort(np.abs(shap_values))[::-1][:top_k]

        shap_values_filtered = shap_values[important_indices]
        X_filtered = X.iloc[:, important_indices]
        
        st.pyplot(shap.plots.force(expected_value, shap_values_filtered, X_filtered, matplotlib=True))

    else:
        st.error("Error: " + response.json().get('error', 'Something went wrong'))