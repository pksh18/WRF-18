import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# Assume `rainfall_data` is your 1D array of daily rainfall measurements
rainfall_data = np.array([10.2, 12.3, 8.5, 6.7, 14.8, 15.6, 11.1, 9.4, 10.5, 13.0, 
                          16.3, 7.8, 6.5, 12.1, 9.0, 10.7, 14.9, 11.2, 8.6, 13.4])  # Example data

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
rainfall_scaled = scaler.fit_transform(rainfall_data.reshape(-1, 1))

# Prepare the data for LSTM
def create_dataset(data, look_back=10):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 10
X, y = create_dataset(rainfall_scaled, look_back)

# Reshape X for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into training and validation sets
split_ratio = 0.8
train_size = int(len(X) * split_ratio)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Define a more complex LSTM model
model = Sequential()

# Add a Bidirectional LSTM layer
model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(X.shape[1], 1)))

# Add a Dropout layer for regularization
model.add(Dropout(0.3))

# Add another LSTM layer
model.add(LSTM(100, return_sequences=False))

# Add another Dropout layer for regularization
model.add(Dropout(0.3))

# Final Dense layer for output
model.add(Dense(1))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# List to store loss values for each epoch
epoch_losses = []

# Custom callback to capture loss at each epoch
class LossHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        epoch_losses.append(logs['loss'])

# Train the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=16, validation_data=(X_val, y_val), verbose=1, callbacks=[LossHistory()])

# Create a DataFrame to display the loss values
loss_df = pd.DataFrame({
    'Epoch': range(1, len(epoch_losses) + 1),
    'Training Loss': epoch_losses,
    'Validation Loss': history.history['val_loss']
})

# Display the table of loss values
print(loss_df)

# Optionally, save the DataFrame to a CSV file
loss_df.to_csv('loss_by_epoch.csv', index=False)
print("Loss data saved to 'loss_by_epoch.csv'.")

# Plot the training and validation loss
plt.figure(figsize=(14, 7))
plt.plot(loss_df['Epoch'], loss_df['Training Loss'], label='Training Loss')
plt.plot(loss_df['Epoch'], loss_df['Validation Loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict the next 10 days
last_10_days = rainfall_scaled[-look_back:].reshape((1, look_back, 1))
predicted_10_days = []
for _ in range(10):
    pred = model.predict(last_10_days)
    predicted_10_days.append(pred[0, 0])  # Append the scalar prediction
    last_10_days = np.append(last_10_days[:, 1:, :], [[pred[0]]], axis=1)

# Convert predicted_10_days to the correct shape for inverse transform
predicted_10_days = np.array(predicted_10_days).reshape(-1, 1)

# Inverse transform to get actual rainfall values
predicted_10_days = scaler.inverse_transform(predicted_10_days).flatten()

# Plot the original data, last 10 days, and predicted values
plt.figure(figsize=(14, 7))

# Plot historical data
plt.plot(rainfall_data, label="Historical Data", color='blue')

# Plot the last 10 days used for prediction
plt.plot(np.arange(len(rainfall_data) - look_back, len(rainfall_data)), 
         rainfall_data[-look_back:], label="Last 10 Days", color='orange', linestyle='dashed')

# Plot the predicted next 10 days
plt.plot(np.arange(len(rainfall_data), len(rainfall_data) + 10), 
         predicted_10_days, label="Predicted Next 10 Days", color='red')

plt.title("Rainfall Prediction")
plt.xlabel("Days")
plt.ylabel("Rainfall")
plt.legend()
plt.show()

# Evaluate the model on the validation set
y_pred_val = model.predict(X_val)
val_mse = mean_squared_error(y_val, y_pred_val)
val_mae = mean_absolute_error(y_val, y_pred_val)
print(f'Validation MSE: {val_mse}')
print(f'Validation MAE: {val_mae}')
