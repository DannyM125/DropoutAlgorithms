import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU

# Load the data
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Age groups to analyze
age_groups = ["Age 16", "Age 17", "Age 18", "Age 19", "Ages  20–24 "]

# Prepare the data for training and testing
years = np.array([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022])  # Missing 2020
X_train = years.reshape(-1, 1)  # Use all years for training
X_test = np.array([[2023]])  # Predict for the year 2023

# Normalize the features (years)
X_scaler = MinMaxScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Function to build and train LSTM model
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to build and train GRU model
def build_gru_model():
    model = Sequential()
    model.add(GRU(50, activation='tanh', input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Loop through each age group
for age_group in age_groups:
    # Check if the age group exists in the data
    if age_group in data.iloc[:, 0].values:
        # Extract the relevant row
        age_data = data[data.iloc[:, 0] == age_group].iloc[0, 1:].astype(str).str.replace(',', '').astype(float).values

        # Prepare target data
        y_train = age_data

        # Normalize the target values
        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))

        # Train and predict with LSTM
        lstm_model = build_lstm_model()
        lstm_model.fit(X_train_lstm, y_train_scaled, epochs=200, verbose=0)
        predicted_2023_lstm_scaled = lstm_model.predict(X_test_lstm)
        predicted_2023_lstm = y_scaler.inverse_transform(predicted_2023_lstm_scaled.reshape(-1, 1))

        # Train and predict with GRU
        gru_model = build_gru_model()
        gru_model.fit(X_train_lstm, y_train_scaled, epochs=200, verbose=0)
        predicted_2023_gru_scaled = gru_model.predict(X_test_lstm)
        predicted_2023_gru = y_scaler.inverse_transform(predicted_2023_gru_scaled.reshape(-1, 1))

        # Train and predict with MLP Regressor
        mlp_regressor = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        mlp_regressor.fit(X_train_scaled, y_train_scaled.ravel())
        predicted_2023_mlp_scaled = mlp_regressor.predict(X_test_scaled)
        predicted_2023_mlp = y_scaler.inverse_transform(predicted_2023_mlp_scaled.reshape(-1, 1))

        # Train and predict with SVR
        svr_model = SVR(kernel='rbf')
        svr_model.fit(X_train_scaled, y_train_scaled.ravel())
        predicted_2023_svr_scaled = svr_model.predict(X_test_scaled)
        predicted_2023_svr = y_scaler.inverse_transform(predicted_2023_svr_scaled.reshape(-1, 1))

        # Print the predictions for 2023
        print(f"\nPredictions for {age_group} in 2023:")
        print(f"LSTM: {predicted_2023_lstm[0][0]}")
        print(f"GRU: {predicted_2023_gru[0][0]}")
        print(f"MLP Regressor: {predicted_2023_mlp[0][0]}")
        print(f"SVR: {predicted_2023_svr[0][0]}")
    else:
        print(f"\nAge group '{age_group}' not found in the data.")
