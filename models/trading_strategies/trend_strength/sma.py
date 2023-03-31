import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from models.trading_strategies.trend_strength.evaluation_models import (
    get_model_prediction,
)


def calculate_RSI(data, period=14):
    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_MACD(data, short_period=12, long_period=26, signal_period=9):
    ema_short = data["close"].ewm(span=short_period).mean()
    ema_long = data["close"].ewm(span=long_period).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_period).mean()

    return macd_line, signal_line


data = pd.read_csv(
    r"data\kc\btc\heiken_ashi\kc_btc_30min_ha.csv",
    parse_dates=["time"],
    index_col="time",
)

# Drop the 'color' column
data = data.drop(columns=["color"])

# Calculate RSI and MACD
data["RSI"] = calculate_RSI(data)
data["MACD_line"], data["Signal_line"] = calculate_MACD(data)
data["MACD"] = data["MACD_line"] - data["Signal_line"]

# Feature Engineering
data["EMA_short"] = data["close"].ewm(span=10).mean()
data["EMA_long"] = data["close"].ewm(span=30).mean()
data["SMA_short"] = data["close"].rolling(window=10).mean()
data["SMA_long"] = data["close"].rolling(window=30).mean()
data["trend_strength"] = data["SMA_short"] - data["SMA_long"]
data["ROC"] = data["close"].pct_change()
data["ATR"] = data["high"] - data["low"]

# Calculate target variable (up or down) based on future price movement
look_ahead = 1
data["target"] = np.where(data["close"].shift(-look_ahead) > data["close"], 1, 0)

# Drop rows with missing values (due to moving average calculations)
data.dropna(inplace=True)

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Test each model separately
model_names = ["arima", "exp_smoothing", "lstm", "prophet"]
for model_name in model_names:
    print(f"Testing {model_name} model...")
    y_pred = get_model_prediction(model_name, train_data, test_data)
    if model_name == "lstm":
        y_true = test_data["close"].iloc[: len(y_pred)]  # Adjust the length of y_true
    else:
        y_true = test_data["close"]
    mse = mean_squared_error(y_true, y_pred)
    print(f"{model_name} MSE:", mse)

    # Convert continuous predictions to binary predictions
    binary_pred = np.where(np.diff(y_pred) > 0, 1, 0)
    binary_true = test_data["target"].iloc[: len(binary_pred)]

    # Calculate the accuracy score
    accuracy = accuracy_score(binary_true, binary_pred)
    print(f"{model_name} Accuracy:", accuracy)
