
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from alpha_vantage.timeseries import TimeSeries
import time


# User Inputs
exchange = input("Enter exchange (NSE/BSE): ").strip().upper()
stock_name = input("Enter stock name: ").strip().upper()
interval = input("Enter timeframe (e.g., 1d, 5min): ").strip()

# Convert to Alpha Vantage format
if exchange == "NSE":
    symbol = f"{stock_name}.NS"  # NSE format
elif exchange == "BSE":
    symbol = f"{stock_name}.BO"  # BSE format
else:
    print("❌ Invalid Exchange. Use 'NSE' or 'BSE'.")
    exit()

# Initialize Alpha Vantage API
API_KEY = "JA8LGGRXPBABCFOA"
ts = TimeSeries(key=API_KEY, output_format="pandas")

# Fetch data with retries
for attempt in range(2):
    try:
        if interval == "1d":
            data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")
        else:
            data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize="full")

        # If the API returned empty data, retry
        if data.empty:
            raise ValueError("No data received. Retrying...")

        break  # Exit loop if successful
    except ValueError as e:
        print(f"⚠ API Error: {e} Retrying in 60 seconds...")
        time.sleep(60)
else:
    print("❌ Failed to retrieve data after multiple attempts.")
    exit()

# Rename columns
data.columns = ["Open", "High", "Low", "Close", "Volume"]

# Print sample data
print("✅ Downloaded Data:")
print(data.head())

# Save to CSV
data.to_csv("data.csv")
print("💾 Data saved to 'data.csv'.")

# Load the saved data
data = pd.read_csv("data.csv", index_col=0, parse_dates=True)

# ✅ Sort data by date to ensure latest data is at the bottom
data = data.sort_index(ascending=True)

# ✅ Check first and last available dates
print(f"📅 First Date in Dataset: {data.index[0]}")
print(f"📅 Last Date in Dataset: {data.index[-1]}")  # Should be recent (2024-2025)


# Convert necessary columns to numeric
for column in ["Open", "High", "Low", "Close", "Volume"]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# Step 5: Calculate indicators
data['Daily Return'] = data['Close'].pct_change()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# Step 6: Handle NaN values by forward filling
data.fillna(method='ffill', inplace=True)

# Step 7: Initialize scalers
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

# Step 8: Scale feature columns
scaled_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
scaled_features = scaler_features.fit_transform(data[scaled_columns])

# Step 9: Scale target column separately
scaled_target = scaler_target.fit_transform(data[['Close']])

# Step 10: Convert scaled features back into a DataFrame
scaled_features_df = pd.DataFrame(scaled_features, columns=scaled_columns, index=data.index)

# Step 11: Merge scaled features with calculated columns
final_data = pd.concat([scaled_features_df, data[['Daily Return', 'SMA_20', 'SMA_50']]], axis=1)

# Print final dataset
print(final_data.tail())


# Step 12: Create sequences for LSTM
def create_sequences(data, target, sequence_length):
    x, y = [], []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i])  # Collect past sequence_length rows
        y.append(target[i])  # Predict the corresponding target
    return np.array(x), np.array(y)

# Set the sequence length
sequence_length = 60

# Prepare sequences
x, y = create_sequences(scaled_features, scaled_target, sequence_length)

# Split data into training and testing sets (80-20 split)
train_size = int(0.8 * len(x))
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 13: Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, x_train.shape[2])),
    Dropout(0.2),  # Prevent overfitting
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),  # Dense layer for non-linear transformations
    Dense(1)  # Output layer to predict the 'Close' price
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32)

# Step 14: Evaluate the model
y_pred_scaled = model.predict(x_test)  # Predict on the scaled test data

# Inverse transform predictions and actual values for evaluation
y_pred = scaler_target.inverse_transform(y_pred_scaled)  # Inverse transform predictions
y_actual = scaler_target.inverse_transform(y_test)  # Inverse transform actual values

'''import pandas as pd
import matplotlib.pyplot as plt'''

# Compute Moving Average and Standard Deviation
window=int(input("Enter Moving average window:"))


data['SMA'] = data['Close'].rolling(window=window).mean()
data['STD'] = data['Close'].rolling(window=window).std()

# Compute Bollinger Bands
data['Upper Band'] = data['SMA'] + (2 * data['STD'])
data['Lower Band'] = data['SMA'] - (2 * data['STD'])

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(data['SMA'], label='SMA (20)', color='orange')
plt.plot(data['Upper Band'], label='Upper Band', color='green')
plt.plot(data['Lower Band'], label='Lower Band', color='red')
plt.fill_between(data.index, data['Lower Band'], data['Upper Band'], color='grey', alpha=0.1)
plt.title('Bollinger Bands')
plt.legend()
plt.show()

# Generate trading signals
data['Signal'] = 'Hold'
data.loc[data['Close'] < data['Lower Band'], 'Signal'] = 'Buy'  # Buy signal
data.loc[data['Close'] > data['Upper Band'], 'Signal'] = 'Sell'  # Sell signal

# Get the last row for the most recent prediction
last_row = data.iloc[-1]
last_date = data.index[-1]  # Use index instead of 'Date' column
last_close = last_row['Close']
last_signal = last_row['Signal']

# Print the prediction in text format
print(f"Date: {last_date}")
print(f"Last Closing Price: {last_close}")
print(f"Prediction: {last_signal}") 


# Step 1: Calculate MACD Components
data['Short_EMA'] = data['Close'].ewm(span=12, adjust=False).mean()  # 12-day EMA
data['Long_EMA'] = data['Close'].ewm(span=26, adjust=False).mean()   # 26-day EMA
data['MACD'] = data['Short_EMA'] - data['Long_EMA']  # MACD Line
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()  # Signal Line
data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']  # MACD Histogram

# Step 2: Compute RSI for momentum confirmation
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['RSI'] = calculate_rsi(data)

# Step 3: Calculate ADX for trend strength
def calculate_adx(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift(1))
    low_close = np.abs(data['Low'] - data['Close'].shift(1))
    tr = high_low.combine(high_close, np.maximum).combine(low_close, np.maximum)

    data['ATR'] = tr.rolling(window=window).mean()

    data['+DM'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']), 
                           data['High'] - data['High'].shift(1), 0)
    data['-DM'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)), 
                           data['Low'].shift(1) - data['Low'], 0)

    data['+DI'] = 100 * (data['+DM'].ewm(span=window, adjust=False).mean() / data['ATR'])
    data['-DI'] = 100 * (data['-DM'].ewm(span=window, adjust=False).mean() / data['ATR'])

    data['DX'] = 100 * (np.abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI']))
    return data['DX'].ewm(span=window, adjust=False).mean()

data['ADX'] = calculate_adx(data)

# Step 4: Define MACD + ADX Trading Strategy
data['MACD_Signal'] = 'Hold'

for i in range(1, len(data)):
    # Buy Conditions:
    if (data['MACD'].iloc[i] > data['Signal_Line'].iloc[i]) and (data['MACD'].iloc[i-1] <= data['Signal_Line'].iloc[i-1]):
        if data['MACD'].iloc[i] > 0 and data['RSI'].iloc[i] > 70 and data['ADX'].iloc[i] > 25:  # Trend & momentum confirmation
            data.at[data.index[i], 'MACD_Signal'] = 'Buy'

    # Sell Conditions:
    elif (data['MACD'].iloc[i] < data['Signal_Line'].iloc[i]) and (data['MACD'].iloc[i-1] >= data['Signal_Line'].iloc[i-1]):
        if data['MACD'].iloc[i] < 0 and data['RSI'].iloc[i] < 70 and data['ADX'].iloc[i] > 25:  # Trend & momentum confirmation
            data.at[data.index[i], 'MACD_Signal'] = 'Sell'

# Step 5: Plot MACD, ADX, and Buy/Sell Signals
plt.figure(figsize=(14, 10))

# Subplot 1: Close Price & Buy/Sell signals
plt.subplot(3, 1, 1)
plt.plot(data['Close'], label='Close Price', color='blue')
plt.scatter(data.index[data['MACD_Signal'] == 'Buy'], data['Close'][data['MACD_Signal'] == 'Buy'], label='Buy Signal', marker='^', color='green', alpha=1)
plt.scatter(data.index[data['MACD_Signal'] == 'Sell'], data['Close'][data['MACD_Signal'] == 'Sell'], label='Sell Signal', marker='v', color='red', alpha=1)
plt.title('Stock Price & MACD + ADX Trading Signals')
plt.legend()

# Subplot 2: MACD Indicator
plt.subplot(3, 1, 2)
plt.plot(data['MACD'], label='MACD', color='green')
plt.plot(data['Signal_Line'], label='Signal Line', color='red')
plt.bar(data.index, data['MACD_Histogram'], label='MACD Histogram', color='gray', alpha=0.3)
plt.legend()

# Subplot 3: ADX Indicator
plt.subplot(3, 1, 3)
plt.plot(data['ADX'], label='ADX', color='black')
plt.axhline(y=25, color='gray', linestyle='--', label='Trend Strength Threshold')
plt.legend()
plt.title('ADX Indicator (Trend Strength)')

plt.show()

# Print the latest MACD signal
last_row = data.iloc[-1]
print(f"Latest MACD Signal: {last_row['MACD_Signal']}")


# Step 1: Calculate Short-Term Support & Resistance
window = 20  # 20-day window for short-term analysis
data['Short_Resistance'] = data['High'].rolling(window=window).max()  # 20-day High
data['Short_Support'] = data['Low'].rolling(window=window).min()  # 20-day Low

# Step 2: Calculate Long-Term Support & Resistance (Pivot Points)
data['Pivot'] = (data['High'].shift(1) + data['Low'].shift(1) + data['Close'].shift(1)) / 3
data['Resistance1'] = (2 * data['Pivot']) - data['Low'].shift(1)
data['Support1'] = (2 * data['Pivot']) - data['High'].shift(1)

# Plotting Support & Resistance Levels
plt.figure(figsize=(14, 7))

# Plot Close Price
plt.plot(data['Close'], label='Close Price', color='blue')

# Plot Short-Term Support & Resistance
plt.plot(data['Short_Support'], label='Short-Term Support (20-day Low)', linestyle='--', color='green')
plt.plot(data['Short_Resistance'], label='Short-Term Resistance (20-day High)', linestyle='--', color='red')

# Plot Long-Term Support & Resistance (Pivot Points)
plt.plot(data['Support1'], label='Long-Term Support (Pivot)', linestyle=':', color='green')
plt.plot(data['Resistance1'], label='Long-Term Resistance (Pivot)', linestyle=':', color='red')

# Add Titles and Legends
plt.title('Support & Resistance Levels')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Find Recent Support & Resistance Levels
recent_support = data['Short_Support'].iloc[-1]
recent_resistance = data['Short_Resistance'].iloc[-1]
long_term_support = data['Support1'].iloc[-1]
long_term_resistance = data['Resistance1'].iloc[-1]

print(f"Recent Short-Term Support: {recent_support}")
print(f"Recent Short-Term Resistance: {recent_resistance}")
print(f"Recent Long-Term Support (Pivot): {long_term_support}")
print(f"Recent Long-Term Resistance (Pivot): {long_term_resistance}")