from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the XGBoost model
# aqi_model_path = r'd:\flut\sus\server\models\xgboost_model_aqi.pkl'
aqi_model_path = './models/xgboost_model_aqi.pkl'
with open(aqi_model_path, 'rb') as file:
    aqimodel = pickle.load(file)

# wqi_model_path = r'd:\flut\sus\server\models\xgboost_model.pkl'
wqi_model_path = './models/xgboost_model.pkl'
with open(wqi_model_path, 'rb') as file:
    wqimodel = pickle.load(file)

# AQI Category Mapping
aqi_categories = {
    0: "Good",
    1: "Moderate",
    2: "Poor",
    3: "Satisfactory",
    5: "Very Poor",
}

# WQI Category Mapping
wqi_categories = {
    0: "Good",
    1: "Fair",
    2: "Marginal",
    3: "Poor",
}
@app.route('/predict-aqi', methods=['POST'])
def predictAQI():
    # Get input data from the request
    data = request.json
    input_df = pd.DataFrame([data])
    print(input_df)
    
    # Predict the AQI category
    prediction = aqimodel.predict(input_df)
    predicted_category = prediction[0]
    
    # Map the prediction to its descriptive category
    predicted_aqi_category = aqi_categories.get(predicted_category, "Unknown Category")
    
    #Return the prediction
    return jsonify({
        'predicted_category': int(predicted_category),
        'predicted_aqi_category': predicted_aqi_category
    })

@app.route('/predict-wqi', methods=['POST'])
def predictWQI():
    # Get input data from the request
    data = request.json
    input_df = pd.DataFrame([data])
    print(input_df)
    
    # Predict the WQI category
    prediction = wqimodel.predict(input_df)
    predicted_category = prediction[0]
    
    # Map the prediction to its descriptive category
    predicted_wqi_category = wqi_categories.get(predicted_category, "Unknown Category")
    
    #Return the prediction
    return jsonify({
        'predicted_category': int(predicted_category),
        'predicted_wqi_category': predicted_wqi_category
    })


    
if __name__ == '__main__':
     # Defaulting to 80 if PORT not found
   app.run(host='0.0.0.0', port=8080)