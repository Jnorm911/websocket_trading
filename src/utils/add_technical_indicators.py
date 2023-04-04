import os
import pandas as pd
import pandas_ta as ta


# Function to add MACD columns to a DataFrame
def add_macd_columns(df):
    macd = ta.macd(df["close"])
    df = pd.concat([df, macd], axis=1)
    return df


# Function to add Standard Pivot Points columns to a DataFrame
def add_standard_pivot_points(df):
    df["PP"] = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
    df["R1"] = 2 * df["PP"] - df["low"].shift(1)
    df["S1"] = 2 * df["PP"] - df["high"].shift(1)
    df["R2"] = df["PP"] + (df["high"].shift(1) - df["low"].shift(1))
    df["S2"] = df["PP"] - (df["high"].shift(1) - df["low"].shift(1))
    df["R3"] = df["high"].shift(1) + 2 * (df["PP"] - df["low"].shift(1))
    df["S3"] = df["low"].shift(1) - 2 * (df["high"].shift(1) - df["PP"])
    return df


# Function to add Bollinger Bands columns to a DataFrame
def add_bollinger_bands(df):
    bollinger = ta.bbands(df["close"])
    df = pd.concat([df, bollinger], axis=1)
    return df


# Function to add RSI column to a DataFrame
def add_rsi(df):
    rsi = ta.rsi(df["close"], length=14)
    df["RSI"] = rsi
    return df


# Function to add SMA columns to a DataFrame
def add_sma(df):
    sma_5 = ta.sma(df["close"], length=5)
    sma_10 = ta.sma(df["close"], length=10)
    sma_20 = ta.sma(df["close"], length=20)

    df["SMA_5"] = sma_5
    df["SMA_10"] = sma_10
    df["SMA_20"] = sma_20

    return df


def add_ema(df):
    ema_5 = ta.ema(df["close"], length=5)
    ema_10 = ta.ema(df["close"], length=10)
    ema_20 = ta.ema(df["close"], length=20)

    df["EMA_5"] = ema_5
    df["EMA_10"] = ema_10
    df["EMA_20"] = ema_20

    return df


def add_stochastic_oscillator(df):
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df = pd.concat([df, stoch], axis=1)
    return df


def add_atr(df):
    atr = ta.atr(df["high"], df["low"], df["close"])
    df["ATR"] = atr
    return df


def add_roc(df):
    roc = ta.roc(df["close"])
    df["ROC"] = roc
    return df


def add_cci(df):
    cci = ta.cci(df["high"], df["low"], df["close"])
    df["CCI"] = cci
    return df


# Directory containing the CSV files
csv_directory = "data/kc/btc/heiken_ashi"

# Directory to save the updated CSV files
output_directory = "data/kc/btc/heiken_ashi/with_trade_indicators"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through each file in the directory
for file_name in os.listdir(csv_directory):
    if file_name.endswith(".csv"):
        file_path = os.path.join(csv_directory, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Add the following columns
        df = add_macd_columns(df)
        df = add_standard_pivot_points(df)
        df = add_bollinger_bands(df)
        df = add_rsi(df)
        df = add_sma(df)
        df = add_ema(df)
        df = add_stochastic_oscillator(df)
        df = add_atr(df)
        df = add_roc(df)
        df = add_cci(df)

        # Create a new file name with '_ti' appended before the file extension
        new_file_name = file_name[:-4] + "_ti.csv"
        new_file_path = os.path.join(output_directory, new_file_name)

        # Save the updated DataFrame to the new file path
        df.to_csv(new_file_path, index=False)
        print(
            f"Updated {file_name} with short to medium length trade indicator columns and saved as {new_file_name} in {output_directory}"
        )
