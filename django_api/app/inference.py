import torch
#import torch.nn as nn
import pandas as pd
import joblib
from datetime import datetime
import os 
from .locationclassifier import Classifier, BIAS, TYPE_MAP, SEASON_DICT, cyclical_encode

NUM_FEATURES = 14

dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"Working directory: {dir_path}")
scaler = joblib.load("scaler.gz")

"""
Passes in data to the neural net to obtain a prediction.
"""
def predict(model, in_features):
    model.eval()
    # Ensure input is properly formatted
    if isinstance(in_features, pd.Series):
        in_features = in_features.to_frame().T
    
    # Convert to tensor
    X = torch.FloatTensor(in_features.values)

    # Make prediction
    with torch.no_grad():
        output = model(X)
        probabilities = torch.softmax(output, dim=1)
        probabilities_lst = probabilities.tolist()
        print(f"probabilities_lst is {probabilities_lst}")
        formatted_probabilities = []

        # for probability in probabilities_lst[0]:
        formatted_probabilities = [x + BIAS[idx] for idx, x in enumerate(probabilities_lst[0])]

        #_, predicted = torch.max(probabilities, 1)
        predicted = formatted_probabilities.index(max(formatted_probabilities))

    # Convert prediction back to label
    quadrant_map = {
        0: 'Center (Kerr Hall, Quad, etc.)',
        1: 'Northwest',
        2: 'Northeast',
        3: 'Southwest',
        4: 'Southeast'
    }
    
    predicted_quadrant = quadrant_map[predicted]
    
    return predicted_quadrant, probabilities.numpy() # returns the raw probabilities

# Example usage:
def make_prediction(model, incident_type, date=datetime.now(), scaler=scaler):
    incident_type = incident_type.lower()
    incident_type = incident_type.replace(" ", "-")
    bin_incident_type = TYPE_MAP[incident_type]
    
    month_sin, month_cos = cyclical_encode(date.month, 12)
    hour_sin, hour_cos = cyclical_encode(date.hour, 24)
    day_of_week_sin, day_of_week_cos = cyclical_encode(date.weekday(), 7)
    weekend = 1 if date.weekday() >= 5 else 0
    season = SEASON_DICT[date.month]

    # Create an incident to represent all of the features 
    sample_features = {
        'incidenttype_High': 0,
        'incidenttype_Low': 0,
        'incidenttype_Med': 0,
        'day_of_week_sin': day_of_week_sin,
        'day_of_week_cos': day_of_week_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'weekend': weekend,
        'season_Fall': 0,
        'season_Spring': 0,
        'season_Summer': 0,
        'season_Winter': 0
    }

    # Setting the incident type
    sample_features[f'incidenttype_{bin_incident_type}'] = 1
    # Setting the season value
    sample_features[f'season_{season}'] = 1
    
    # Convert to DataFrame
    input_df = pd.DataFrame([sample_features])
    
    # Scale the numerical features (the ones that were scaled during training)
    numerical_columns = [col for col in input_df.columns 
                        if col.endswith(('_sin', '_cos'))]
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
    
    # Make prediction
    predicted_quadrant, probabilities = predict(model, input_df)
    
    print(f"Predicted Quadrant: {predicted_quadrant}")
    print("\nProbabilities for each quadrant:")
    for quadrant, prob in zip(['Center', 'NW', 'NE', 'SW', 'SE'], probabilities[0]):
        print(f"{quadrant}: {prob:.4f}")

    return predicted_quadrant

if __name__ == "__main__":
    model = Classifier(input_size=NUM_FEATURES)
    # model.load_state_dict(torch.load("models/model_20241123_143841_4"))
    # model.load_state_dict(torch.load("models/model_20241123_163146_8"))
    model.load_state_dict(torch.load("models/model_20241123_183614_10"))
    # Use the model
    model.eval()  # Set to evaluation mode
    # Loading the scaler object used during training
    scaler = joblib.load("scaler.gz")
    make_prediction(model, scaler, incident_type="Robbery")  # scaler is the StandardScaler used during training