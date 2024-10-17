import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import plotly.graph_objects as go

# Step 1: Load and Clean the Dataset
data = pd.read_csv("C:/Users/ccape/Downloads/daily-minimum-temperatures-in-me.csv")

# Use the first column as the target (by index)
target_column = data.columns[1]  # Adjust to target the correct column by index

# Ensure the target column is numeric and clean NaN values
data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
data.dropna(inplace=True)

# Convert 'Date' to datetime format if available and set it as index
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

# Step 2: Normalize the Target Data
scaler = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler.fit_transform(data[[target_column]])

# Step 3: Create Input Sequences (30 days to predict the next day)
def create_dataset(dataset, time_step=30):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:i + time_step, 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

X, Y = create_dataset(target_scaled)

# Reshape data for LSTM input
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 4: Split the Data (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Step 5: Build the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# Step 6: Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 7: Train the Model
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))

# Step 8: Make Predictions
Y_pred = model.predict(X_test)
Y_pred_inverse = scaler.inverse_transform(Y_pred)

# Step 9: Evaluate the Model
Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
mae = mean_absolute_error(Y_test_inverse, Y_pred_inverse)
mse = mean_squared_error(Y_test_inverse, Y_pred_inverse)
rmse = math.sqrt(mse)

print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

# Step 10: Save Predictions and Accuracy to CSV
results_df = pd.DataFrame({
    'Date': data.index[-len(Y_test):], 
    'Actual Value': Y_test_inverse.flatten(), 
    'Predicted Value': Y_pred_inverse.flatten()
})
results_df['Absolute Error'] = abs(results_df['Actual Value'] - results_df['Predicted Value'])

# Save the results to CSV
results_df.to_csv("LSTM_model_output.csv", index=False)
print("Results saved to LSTM_model_output.csv")

# Step 11: Plot Predictions vs Actual Values with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Actual Value'], mode='lines', name='Actual Value'))
fig.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Predicted Value'], mode='lines', name='Predicted Value'))

# Customize the layout
fig.update_layout(
    title='LSTM Forecasting',
    xaxis_title='Date',
    yaxis_title='Value',
    legend=dict(x=0, y=1, traceorder='normal'),
    template='plotly_dark'
)

# Show the interactive plot
fig.show()
