import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU

# Load the data
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Categories to analyze
categories = [
    "Spoke English at home or spoke English very well ",
    "Spoke a language other than English at home and spoke English less than very well "
]

# Prepare the data for training and testing
years = np.array([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021])  # Training set (2012-2021, missing 2020)
years_test = np.array([2022])  # Test set (2022)
X_train = years.reshape(-1, 1)
X_test = years_test.reshape(-1, 1)

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

# Loop through each category
for category in categories:
    # Check if the category exists in the data
    if category in data.iloc[:, 0].values:
        # Extract the relevant row
        category_data = data[data.iloc[:, 0] == category].iloc[0, 1:].astype(str).str.replace(',', '').astype(float).values

        # Split data into training and testing sets
        y_train = category_data[:-1]
        y_test = category_data[-1]
        
        # Normalize the target values
        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))

        # Train and predict with LSTM
        lstm_model = build_lstm_model()
        lstm_model.fit(X_train_lstm, y_train_scaled, epochs=200, verbose=0)
        predicted_2022_lstm_scaled = lstm_model.predict(X_test_lstm)
        predicted_2022_lstm = y_scaler.inverse_transform(predicted_2022_lstm_scaled.reshape(-1, 1))

        # Train and predict with GRU
        gru_model = build_gru_model()
        gru_model.fit(X_train_lstm, y_train_scaled, epochs=200, verbose=0)
        predicted_2022_gru_scaled = gru_model.predict(X_test_lstm)
        predicted_2022_gru = y_scaler.inverse_transform(predicted_2022_gru_scaled.reshape(-1, 1))

        # Train and predict with MLP Regressor
        mlp_regressor = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        mlp_regressor.fit(X_train_scaled, y_train_scaled.ravel())
        predicted_2022_mlp_scaled = mlp_regressor.predict(X_test_scaled)
        predicted_2022_mlp = y_scaler.inverse_transform(predicted_2022_mlp_scaled.reshape(-1, 1))

        # Train and predict with SVR
        svr_model = SVR(kernel='rbf')
        svr_model.fit(X_train_scaled, y_train_scaled.ravel())
        predicted_2022_svr_scaled = svr_model.predict(X_test_scaled)
        predicted_2022_svr = y_scaler.inverse_transform(predicted_2022_svr_scaled.reshape(-1, 1))

        # Calculate MSE for all models
        mse_lstm = mean_squared_error([y_test], predicted_2022_lstm)
        mse_gru = mean_squared_error([y_test], predicted_2022_gru)
        mse_mlp = mean_squared_error([y_test], predicted_2022_mlp)
        mse_svr = mean_squared_error([y_test], predicted_2022_svr)

        # Print the predictions for 2022 and their MSE
        print(f"\nPredictions for {category} in 2022:")
        print(f"LSTM: {predicted_2022_lstm[0][0]} - MSE: {mse_lstm}")
        print(f"GRU: {predicted_2022_gru[0][0]} - MSE: {mse_gru}")
        print(f"MLP Regressor: {predicted_2022_mlp[0][0]} - MSE: {mse_mlp}")
        print(f"SVR: {predicted_2022_svr[0][0]} - MSE: {mse_svr}")
    else:
        print(f"\nCategory '{category}' not found in the data.")
