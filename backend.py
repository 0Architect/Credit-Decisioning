from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model (ensure the model is saved as 'credit_model.pkl')
model = joblib.load('random_forest_model.pkl')  # Replace with your model path

# Create a Flask app
app = Flask(__name__)

# Load scaler if necessary (if you used scaling during training)
scaler = joblib.load('scaler.pkl')  # Replace with the actual scaler file, if applicable

# Function to process input data (convert to DataFrame for prediction)
def process_input(data):
    # Convert the input into a DataFrame (this is necessary for model prediction)
    input_data = pd.DataFrame([data])
    
    # Optionally: Scale the features if you scaled them during training
    input_data_scaled = scaler.transform(input_data)
    
    return input_data_scaled

@app.route('/predict', methods=['POST'])
def predict_credit_score():
    try:
        # Get the data from the POST request
        data = request.get_json()

        # Validate if the required fields are present in the input
        required_fields = [
            'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 
            'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
            'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Process the input data
        processed_data = process_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)

        # Return the prediction as a JSON response
        result = {"credit_score": int(prediction[0])}
        return jsonify(result)
    
    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)