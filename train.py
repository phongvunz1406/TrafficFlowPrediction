"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import os
warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """


    # model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    #Use Adam optimizer with initial learning rate 0.001
    model.compile(loss="mse", optimizer="adam", metrics=['mape'])
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-5, verbose=1)


    early = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')#Early stopping
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        callbacks = [early,lr_reduction]#Add early stopping & learning rate reduction
        )

    model.save('model/' + name + '.keras')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)


def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    early = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(p.inputs,
                                       p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05,
              callbacks = [early]
              )

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", help="Model to train.")
    args = parser.parse_args()

    data_dir = "data/split_data"  # Adjust this path
    lag = 12
    config = {"batch": 64, "epochs": 600}

    # Initialize empty lists to store data
    X_train_list = []
    y_train_list = []

    # Loop through all train files and process corresponding test files
    for file in os.listdir(data_dir):
        if file.startswith("train_") and file.endswith(".csv"):
            location_name = file.replace("train_", "").replace(".csv", "")
            train_file = os.path.join(data_dir, file)
            test_file = os.path.join(data_dir, f"test_{location_name}.csv")

            if not os.path.exists(test_file):
                print(f"Warning: Test file {test_file} not found. Skipping this location.")
                continue

            # Process the data
            X_train, y_train, _, _, _ = process_data(train_file, test_file, lag)

            # Store the processed data
            X_train_list.append(X_train)
            y_train_list.append(y_train)

    # Combine all train data
    if len(X_train_list) == 0:
        print("No training data found. Exiting...")
        return

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # Reshape input for LSTM/GRU if needed
    if args.model in ['lstm', 'gru']:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Train model based on selected type
    if args.model == 'lstm':
        m = model.get_lstm([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)

    elif args.model == 'gru':
        m = model.get_gru([12, 128, 256, 1])  #  [12,64,64,1] --------- [12,128,128,1]
        train_model(m, X_train, y_train, args.model, config)

    elif args.model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_saes([12, 400, 400, 400, 1])
        train_seas(m, X_train, y_train, args.model, config)
    elif args.model == 'cnn':
        m = model.get_cnn([12,64,64,1])
        train_model(m, X_train, y_train, args.model, config)
    else:
        print(f"Model {args.model} is not available")
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv)