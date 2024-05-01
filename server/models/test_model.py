


import pickle
import pandas as pd


def load_model(model_path):
    """
    Loads the XGBoost model from the specified path.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def prepare_input_data(sample_values):
    """
    Prepares the input data from the provided sample values into a DataFrame.
    """
    columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
               'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    data = pd.DataFrame([sample_values], columns=columns)
    return data


def predict_aqi(model, input_data):
    """
    Predicts the AQI category using the loaded model and the prepared input data.
    """
    prediction = model.predict(input_data)
    return prediction[0]


# AQI Category Mapping
aqi_categories = {
    0: "Good",
    1: "Moderate",
    2: "Poor",
    3: "Satisfactory",
    5: "Very Poor",
}

# Correct path to the model file
model_path = './models/xgboost_model_aqi.pkl'
model = load_model(model_path)

# Sample values - make sure these are in the correct format and order
sample_values = [122.88,208.86,5.56,54.87,33.71,17.96,0.27,22.97,68.6,0.36,6.28,0.21]
input_data = prepare_input_data(sample_values)

predicted_category_num = predict_aqi(model, input_data)
predicted_aqi_category = aqi_categories.get(predicted_category_num, "Unknown")

print(
    f"Predicted AQI Category: {predicted_category_num} - {predicted_aqi_category}")
