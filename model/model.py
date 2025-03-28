"""
Defination of NN model
"""
from keras.layers import Dense, Dropout, Activation, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential


def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2], return_sequences=False))
    model.add(Dropout(0.3)) #Increase dropout for regularization (0.2 -> 0.3)
    # model.add(Dense(units[3], activation='sigmoid')) Change activation from sigmod to relu
    model.add(Dense(units[3], activation = 'relu'))
    # model.add(Dense(units[3], activation = 'tanh'))
    return model

def get_cnn(units):
    """Build CNN Model (Convolutional Neural Network)
    
    # Arguments:
        units: List(int), number of input, output and hidden units.
               units[0]: input length
               units[1]: first conv layer filters
               units[2]: second conv layer filters
               units[3]: output dimension
    
    # Returns:   
        model: Model, nn model 
    """
    model = Sequential()
    # First convolutional layer
    model.add(Conv1D(filters=units[1], kernel_size=3, activation='relu',
                    padding='same', input_shape=(units[0], 1)))
    model.add(MaxPooling1D(pool_size=2))
    # Second convolutional layer
    model.add(Conv1D(filters=units[2], kernel_size=3, activation='relu',
                    padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    # Flatten layer to connect to dense layer
    model.add(Flatten())
    # Dense layer with dropout for regularization
    model.add(Dense(units[2]))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    # Output layer
    model.add(Dense(units[3], activation='sigmoid'))
    return model



def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='sigmoid'))

    return model


def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.

    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[2], name='hidden2'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[3], name='hidden3'))
    saes.add(Activation('sigmoid'))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation='sigmoid'))

    models = [sae1, sae2, sae3, saes]

    return models