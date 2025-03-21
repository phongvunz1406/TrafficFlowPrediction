import math
import warnings
import numpy as np
import pandas as pd
import os
from data.data import process_data
from keras.models import load_model
from tensorflow.keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.losses import MeanSquaredError
from keras.saving import register_keras_serializable
import pydot
import graphviz

warnings.filterwarnings("ignore")

# Register custom mse function
@register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = sum(abs(y[i] - y_pred[i]) / y[i] for i in range(num))
    
    return sums * (100 / num)

def eva_regress(y_true, y_pred):
    """Evaluate Model Performance"""
    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    
    print(f'explained_variance_score: {vs:.6f}')
    print(f'mape: {mape:.6f}%')
    print(f'mae: {mae:.6f}')
    print(f'mse: {mse:.6f}')
    print(f'rmse: {math.sqrt(mse):.6f}')
    print(f'r2: {r2:.6f}')

def plot_results(y_true, y_preds, names, location):
    """Plot results for a specific location"""
    num_points = len(y_true)  # Dynamically determine the number of points
    d = '2006-10-1 00:00'
    x = pd.date_range(d, periods=num_points, freq='15min')  # Adjust time range

    fig, ax = plt.subplots()
    ax.plot(x, y_true, label='True Data')

    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred[:num_points], label=name)  # Ensure same length

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # Major tick every 2 hours
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))  # Minor tick every 30 mins
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.autofmt_xdate()  # Rotate labels for better readability

    plt.title(f"Predictions for {location}")
    plt.show()

# def main():
#     # Load trained models
#     lstm = load_model('model/lstm.keras', custom_objects={'mse': mse})
#     gru = load_model('model/gru.keras', custom_objects={'mse': mse})
#     saes = load_model('model/saes.keras', custom_objects={'mse': mse})
#     cnn = load_model('model/cnn.keras', custom_objects={'mse': mse})

#     models = [lstm, gru, saes, cnn]
#     names = ['LSTM', 'GRU', 'SAEs', 'CNN']

#     train_file = "data/split_data/train_0970_HIGH_STREET_RD_E_of_WARRIGAL_RD.csv"
#     test_file = "data/split_data/test_0970_HIGH_STREET_RD_E_of_WARRIGAL_RD.csv"
#     location_name = "0970_HIGH_STREET_RD_E_of_WARRIGAL_RD"
#     lag = 12

#     print(f"\nEvaluating models for {location_name}...")

#     # Process the data
#     _, _, X_test, y_test, scaler = process_data(train_file, test_file, lag)
#     y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

#     y_preds = []
#     for name, model in zip(names, models):
#         if name == 'SAEs':
#             X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
#         else:
#             X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#         file_path = f'images/{name}_{location_name}.png'
#         plot_model(model, to_file=file_path, show_shapes=True)

#         predicted = model.predict(X_test_reshaped)
#         predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
#         y_preds.append(predicted[:96])

#         print(f"\n{name} Model Evaluation for {location_name}:")
#         eva_regress(y_test, predicted)

#     plot_results(y_test[:96], y_preds, names, location_name)

def main():
    # Load trained models once
    lstm = load_model('model/lstm.keras', custom_objects={'mse': mse})
    gru = load_model('model/gru.keras', custom_objects={'mse': mse})
    saes = load_model('model/saes.keras', custom_objects={'mse': mse})
    cnn = load_model('model/cnn.keras', custom_objects={'mse': mse})
    models = [lstm, gru, saes, cnn]
    names = ['LSTM', 'GRU', 'SAEs', 'CNN']
    lag = 12

    # Automatically process each unique test file (location)
    test_folder = "data/split_data/"
    test_files = [f for f in os.listdir(test_folder) if f.startswith("test_")]

    for test_file in test_files:
        # Extract location name
        location_name = test_file.replace("test_", "").replace(".csv", "")
        train_file = os.path.join(test_folder, f"train_{location_name}.csv")
        test_file_path = os.path.join(test_folder, test_file)

        # Check if the corresponding train file exists
        if not os.path.exists(train_file):
            print(f"Skipping {location_name}: Train file not found.")
            continue

        print(f"\nEvaluating models for {location_name}...")

        # Process the data
        _, _, X_test, y_test, scaler = process_data(train_file, test_file_path, lag)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

        y_preds = []
        for name, model in zip(names, models):
            if name == 'SAEs':
                X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            else:
                X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # Optional: Save model architecture diagram
            file_path = f'images/{name}_{location_name}.png'
            plot_model(model, to_file=file_path, show_shapes=True)

            predicted = model.predict(X_test_reshaped)
            predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
            y_preds.append(predicted[:96])  # Plot only the first 96 points (1 day)

            print(f"\n{name} Model Evaluation for {location_name}:")
            eva_regress(y_test, predicted)

        # Plot results for this location
        plot_results(y_test[:96], y_preds, names, location_name)
if __name__ == '__main__':
    main()