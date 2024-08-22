import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU


# Load the data
file_path = 'data.csv'
data = pd.read_csv(file_path)


# Extract relevant rows and columns, accounting for the missing year 2020
total_dropouts = data.iloc[0, 1:].str.replace(',', '').astype(float).values
dropout_rate = data.iloc[1, 1:].astype(float).values
years = np.array([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022])  # Missing 2020


# Prepare the data for training and testing
X_train = years.reshape(-1, 1)  # Use all years for training
X_test = np.array([[2023]])  # Predict for the year 2023


y_train_dropouts = total_dropouts
y_train_rate = dropout_rate


# Normalize the features (years) and target values
X_scaler = MinMaxScaler()
y_scaler_dropouts = MinMaxScaler()
y_scaler_rate = MinMaxScaler()


X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


y_train_dropouts_scaled = y_scaler_dropouts.fit_transform(y_train_dropouts.reshape(-1, 1))
y_train_rate_scaled = y_scaler_rate.fit_transform(y_train_rate.reshape(-1, 1))


# Prepare data for LSTM and GRU models
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


# Define and train LSTM model for dropouts
def build_lstm_model():
   model = Sequential()
   model.add(LSTM(50, activation='tanh', input_shape=(1, 1)))
   model.add(Dense(1))
   model.compile(optimizer='adam', loss='mse')
   return model


lstm_model_dropouts = build_lstm_model()
lstm_model_dropouts.fit(X_train_lstm, y_train_dropouts_scaled, epochs=200, verbose=0)


lstm_model_rate = build_lstm_model()
lstm_model_rate.fit(X_train_lstm, y_train_rate_scaled, epochs=200, verbose=0)


# Define and train GRU model for dropouts
def build_gru_model():
   model = Sequential()
   model.add(GRU(50, activation='tanh', input_shape=(1, 1)))
   model.add(Dense(1))
   model.compile(optimizer='adam', loss='mse')
   return model


gru_model_dropouts = build_gru_model()
gru_model_dropouts.fit(X_train_lstm, y_train_dropouts_scaled, epochs=200, verbose=0)


gru_model_rate = build_gru_model()
gru_model_rate.fit(X_train_lstm, y_train_rate_scaled, epochs=200, verbose=0)


# Train MLP Regressor for total dropouts
mlp_regressor_dropouts = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp_regressor_dropouts.fit(X_train_scaled, y_train_dropouts_scaled.ravel())


# Train MLP Regressor for dropout rates
mlp_regressor_rate = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp_regressor_rate.fit(X_train_scaled, y_train_rate_scaled.ravel())


# Train SVR for total dropouts
svr_dropouts = SVR(kernel='rbf')
svr_dropouts.fit(X_train_scaled, y_train_dropouts_scaled.ravel())


# Train SVR for dropout rates
svr_rate = SVR(kernel='rbf')
svr_rate.fit(X_train_scaled, y_train_rate_scaled.ravel())


# Predictions
def make_predictions(model, X):
   return model.predict(X)


# Predict 2023 values with all models
X_test_scaled = X_scaler.transform(X_test)  # Ensure X_test is scaled correctly
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


predicted_2023_dropouts_lstm = make_predictions(lstm_model_dropouts, X_test_lstm)
predicted_2023_rate_lstm = make_predictions(lstm_model_rate, X_test_lstm)


predicted_2023_dropouts_gru = make_predictions(gru_model_dropouts, X_test_lstm)
predicted_2023_rate_gru = make_predictions(gru_model_rate, X_test_lstm)


predicted_2023_dropouts_mlp = mlp_regressor_dropouts.predict(X_test_scaled)
predicted_2023_rate_mlp = mlp_regressor_rate.predict(X_test_scaled)


predicted_2023_dropouts_svr = svr_dropouts.predict(X_test_scaled)
predicted_2023_rate_svr = svr_rate.predict(X_test_scaled)


# Inverse transform predictions
predicted_2023_dropouts_lstm = y_scaler_dropouts.inverse_transform(predicted_2023_dropouts_lstm.reshape(-1, 1))
predicted_2023_rate_lstm = y_scaler_rate.inverse_transform(predicted_2023_rate_lstm.reshape(-1, 1))


predicted_2023_dropouts_gru = y_scaler_dropouts.inverse_transform(predicted_2023_dropouts_gru.reshape(-1, 1))
predicted_2023_rate_gru = y_scaler_rate.inverse_transform(predicted_2023_rate_gru.reshape(-1, 1))


predicted_2023_dropouts_mlp = y_scaler_dropouts.inverse_transform(predicted_2023_dropouts_mlp.reshape(-1, 1))
predicted_2023_rate_mlp = y_scaler_rate.inverse_transform(predicted_2023_rate_mlp.reshape(-1, 1))


predicted_2023_dropouts_svr = y_scaler_dropouts.inverse_transform(predicted_2023_dropouts_svr.reshape(-1, 1))
predicted_2023_rate_svr = y_scaler_rate.inverse_transform(predicted_2023_rate_svr.reshape(-1, 1))


# Print the predictions for 2023
print("\nLSTM Predictions for 2023:")
print(f"Total Dropouts: {predicted_2023_dropouts_lstm[0][0]}")
print(f"Dropout Rate: {predicted_2023_rate_lstm[0][0]}")


print("\nGRU Predictions for 2023:")
print(f"Total Dropouts: {predicted_2023_dropouts_gru[0][0]}")
print(f"Dropout Rate: {predicted_2023_rate_gru[0][0]}")


print("\nMLP Regressor Predictions for 2023:")
print(f"Total Dropouts: {predicted_2023_dropouts_mlp[0]}")
print(f"Dropout Rate: {predicted_2023_rate_mlp[0]}")


print("\nSVR Predictions for 2023:")
print(f"Total Dropouts: {predicted_2023_dropouts_svr[0]}")
print(f"Dropout Rate: {predicted_2023_rate_svr[0]}")
