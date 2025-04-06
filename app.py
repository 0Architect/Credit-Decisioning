from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import shap

model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
explainer = shap.TreeExplainer(model)

app = Flask(__name__)

required_fields = [
'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 
'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
]

def process_input(data):
    input_data = pd.DataFrame([data])
    input_data_scaled = scaler.transform(input_data)
    return input_data, input_data_scaled

@app.route('/predict', methods=['POST'])
def predict_credit_score():
    try:
        data = request.get_json()

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        input_df, processed_data = process_input(data)

        prediction = int(model.predict(processed_data)[0])

        shap_output = explainer(input_df)
        shap_values_class_1 = shap_output.values[0, :, 1]
        expected_value = shap_output.base_values[0][1]

        result = {
            "credit_score": prediction, 
            "shap_values": shap_values_class_1.tolist(),
            "expected_value": expected_value,
            "features": input_df.iloc[0].to_dict(),
            }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)