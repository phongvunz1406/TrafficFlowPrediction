import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# File paths
input_file = "data/Scats Data October 2006.xls"
output_directory = "output_data"
split_directory = "split_data"

def load_data(file_path, sheet_name="Data"):
    """
    Loads and preprocesses the SCATS traffic data from an Excel file.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='xlrd', header=1)
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
    return df

def transform_traffic_data(df):
    """
    Transforms raw traffic data into a structured format with 15-minute intervals.
    """
    transformed_records = []

    for _, row in df.iterrows():
        scats_id = str(int(row['SCATS Number'])).zfill(4)  # Ensure SCATS ID is a 4-digit string
        location = row['Location']
        latitude = row['NB_LATITUDE']
        longitude = row['NB_LONGITUDE']
        date = pd.to_datetime(row['Date'])

        for interval in range(96):  # 96 intervals = 24(hours)*60/15
            interval_col = f"V{interval:02d}"
            time_slot = date + pd.Timedelta(minutes=interval * 15)

            transformed_records.append({
                'SCATS Number': scats_id,
                'Location': location,
                'NB_LATITUDE': latitude,
                'NB_LONGITUDE': longitude,
                '15 Minutes': time_slot,
                'Lane 1 Flow (Veh/15 Minutes)': row.get(interval_col, 0),  
            })

    return pd.DataFrame(transformed_records)

def save_location_data(df, output_dir):
    """
    Saves each unique location's traffic data to separate CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    location_groups = df.groupby('Location')  # Group data by location

    for site_location, group_data in location_groups:
        safe_location = site_location.replace(" ", "_").replace("/", "_").replace("\\", "_")
        output_file_name = f'{group_data["SCATS Number"].iloc[0]}_{safe_location}.csv'
        output_file_path = os.path.join(output_dir, output_file_name)

        group_data.to_csv(output_file_path, index=False)

def split_data(input_folder="output_data", output_folder="split_data"):
    """
    Splits each location's traffic data into train and test datasets (80/20 split).
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output directory if not exists

    for file in os.listdir(input_folder):
        if file.endswith(".csv"):  
            file_path = os.path.join(input_folder, file)

            # Load each location's CSV
            df = pd.read_csv(file_path)

            # Convert '15 Minutes' column to datetime format and sort
            df['15 Minutes'] = pd.to_datetime(df['15 Minutes'])
            df = df.sort_values(by='15 Minutes')  

            # Split data (No shuffle to maintain time order)
            train, test = train_test_split(df, test_size=0.2, shuffle=False)  

            # Save train and test files
            train.to_csv(os.path.join(output_folder, f"train_{file}"), index=False)
            test.to_csv(os.path.join(output_folder, f"test_{file}"), index=False)

def process_data(train, test, lags):
    """
    Processes data by reshaping and splitting train/test data.
    """
    attr = 'Lane 1 Flow (Veh/15 Minutes)'  # Ensure correct column name
    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

    # Use MinMaxScaler for normalization
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train_data, test_data = [], []
    for i in range(lags, len(flow1)):
        train_data.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test_data.append(flow2[i - lags: i + 1])

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    np.random.shuffle(train_data)

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    return X_train, y_train, X_test, y_test, scaler

