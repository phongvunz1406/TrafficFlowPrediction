import os
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from keras.models import load_model
import numpy as np
from data.data import process_data

# Haversine function to calculate distance in km
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distance in km

# Predict traffic flow using trained model
def predict_traffic_flow(location, time, model_type):
    model_paths = {
        "GRU": r"model/gru.keras",
        "LSTM": r"model/lstm.keras",
        "SAES": r"model/saes.keras",
        "CNN": r"model/cnn.keras"
    }

    if model_type not in model_paths:
        print(f"Invalid model '{model_type}', using default GRU model.")
        model_type = "GRU"

    model_path = model_paths[model_type]
    
    try:
        model = load_model(model_path)
    except:
        print(f"Model '{model_type}' not found for {location}, using default traffic value")
        return 1000  # Default flow value if model is missing

    location = location.replace(" ", "_").replace("/", "_").replace("\\", "_")
    test_file = f'data/output_data/{location}.csv'

    if not os.path.exists(test_file):
        print(f" ERROR: Test file '{test_file}' not found!")
        print("Available files:", os.listdir('data/output_data'))  # Debugging line
        return 1000

    _, _, X_test, _, scaler = process_data(test_file, test_file, 12)
    time_index = (pd.to_datetime(time) - pd.to_datetime('2006-10-26 00:00')).seconds // 900
    X_test_nn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    predicted = model.predict(X_test_nn)
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]

    print(f"Predicted traffic flow for {location} at {time} using {model_type}: {predicted[time_index]}")
    return round(predicted[time_index])

# Get user input
def get_user_input(): 
    origin = input("Enter origin: ")
    destination = input("Enter destination: ")
    date_time = input("Enter date and time (MM/DD/YYYY HH:MM:SS AM/PM): ")
    
    # Allow users to choose prediction model
    models = ["GRU", "LSTM", "SAES", "CNN"]
    print(f"Available models: {', '.join(models)}")
    while True:
        choose_model = input("Choose model: ").strip().upper()
        if choose_model in models:
            break
        print("Invalid model! Please select a valid model.")

    return origin, destination, date_time, choose_model

# Estimate travel time
def estimate_travel_time(origin, destination, date_time, choose_model):
    scats_files = [f for f in os.listdir('data/output_data') if f.endswith('.csv')]

    origin_file = next((f for f in scats_files if origin.replace(" ", "_") in f), None)
    destination_file = next((f for f in scats_files if destination.replace(" ", "_") in f), None)

    if not origin_file or not destination_file:
        return "Error: Could not find SCATS data for the given locations.\n"

    # Get coordinates
    origin_lat, origin_lon = get_scats_coordinates(origin_file)
    dest_lat, dest_lon = get_scats_coordinates(destination_file)

    # Compute distance using Haversine
    distance = haversine(origin_lat, origin_lon, dest_lat, dest_lon)

    # Predict traffic flow
    traffic_flow = predict_traffic_flow(origin, date_time, choose_model)

    # Compute speed based on traffic flow
    base_speed = 60  # Free-flow speed (no congestion)
    F_base = max(traffic_flow * 0.5, 10)  # Lower threshold for congestion effects
    lambda_decay = 0.02  # Higher decay rate for speed reduction

    excess_flow = max(traffic_flow - F_base, 0)
    speed = base_speed * np.exp(-lambda_decay * excess_flow)

    # Calculate travel time
    estimated_time_hours = distance / speed
    estimated_time_min = estimated_time_hours * 60

    # Build the output string for GUI
    result = (f"Prediction Result:\n"
              f"From: {origin}\n"
              f"To: {destination}\n"
              f"Distance: {distance:.2f} km\n"
              f"Traffic Flow: {traffic_flow:.2f}\n"
              f"Estimated Speed: {speed:.2f} km/h\n"
              f"Estimated Travel Time: {estimated_time_min:.0f} minutes\n")
    return result


# Get SCATS coordinates from file
def get_scats_coordinates(filename):
    df = pd.read_csv(f'data/output_data/{filename}')
    if 'NB_LONGITUDE' not in df.columns:
        print(f"Column 'NB_LONGITUDE' not found in {filename}. Available columns: {df.columns}")
        return None, None
    return df.iloc[0]['NB_LATITUDE'], df.iloc[0]['NB_LONGITUDE']

# Run program
if __name__ == "__main__":
    origin, destination, date_time, choose_model = get_user_input()
    estimate_travel_time(origin, destination, date_time, choose_model)
