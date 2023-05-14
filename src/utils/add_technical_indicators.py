import os
import pandas as pd
import pandas_ta as ta


def add_macd_columns(df):
    macd = ta.macd(df["close"])
    df = pd.concat([df, macd], axis=1)
    return df


def add_standard_pivot_points(df):
    df["PP"] = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
    df["R1"] = 2 * df["PP"] - df["low"].shift(1)
    df["S1"] = 2 * df["PP"] - df["high"].shift(1)
    df["R2"] = df["PP"] + (df["high"].shift(1) - df["low"].shift(1))
    df["S2"] = df["PP"] - (df["high"].shift(1) - df["low"].shift(1))
    df["R3"] = df["high"].shift(1) + 2 * (df["PP"] - df["low"].shift(1))
    df["S3"] = df["low"].shift(1) - 2 * (df["high"].shift(1) - df["PP"])
    return df


def add_sma(df):
    sma_5 = ta.sma(df["close"], length=5)
    sma_10 = ta.sma(df["close"], length=10)
    sma_2 = ta.sma(df["close"], length=2)

    df["SMA_20"] = sma_2
    df["SMA_5"] = sma_5
    df["SMA_10"] = sma_10

    return df


def add_ema(df):
    ema_2 = ta.ema(df["close"], length=2)
    ema_5 = ta.ema(df["close"], length=5)
    ema_10 = ta.ema(df["close"], length=10)

    df["EMA_2"] = ema_2
    df["EMA_5"] = ema_5
    df["EMA_10"] = ema_10

    return df


def add_stochastic_oscillator(df):
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df = pd.concat([df, stoch], axis=1)
    return df


def add_atr(df):
    atr_14 = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ATR_14"] = atr_14

    atr_10 = ta.atr(df["high"], df["low"], df["close"], length=10)
    df["ATR_10"] = atr_10

    atr_5 = ta.atr(df["high"], df["low"], df["close"], length=5)
    df["ATR_5"] = atr_5

    return df


def add_roc(df):
    roc_14 = ta.roc(df["close"], length=14)
    df["ROC_14"] = roc_14

    roc_10 = ta.roc(df["close"], length=10)
    df["ROC_10"] = roc_10

    roc_5 = ta.roc(df["close"], length=5)
    df["ROC_5"] = roc_5

    return df


def add_cci(df):
    cci_14 = ta.cci(df["high"], df["low"], df["close"], length=14)
    df["CCI_14"] = cci_14

    cci_10 = ta.cci(df["high"], df["low"], df["close"], length=10)
    df["CCI_10"] = cci_10

    cci_5 = ta.cci(df["high"], df["low"], df["close"], length=5)
    df["CCI_5"] = cci_5

    return df


def add_bollinger_bandwidth(df):
    bollinger = ta.bbands(df["close"], length=20)
    df["bollinger_bandwidth"] = (
        bollinger["BBU_20_2.0"] - bollinger["BBL_20_2.0"]
    ) / bollinger["BBM_20_2.0"]
    return df


def add_bollinger_bands(df):
    for length in [5, 10, 15, 20]:
        bollinger = ta.bbands(df["close"], length=length)
        df = pd.concat([df, bollinger.add_suffix(f"_{length}")], axis=1)
    return df


def add_cmf(df):
    df["cmf"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=7)
    return df


def add_obv(df):
    df["obv"] = ta.obv(df["close"], df["volume"])
    return df


def add_psar(df, af0=0.01, af=0.01, max_af=0.1):
    psar_df = ta.psar(df["high"], df["low"], af0=af0, af=af, max_af=max_af)
    df = pd.concat([df, psar_df], axis=1)
    return df


def add_ichimoku(df, tenkan=5, kijun=15, senkou=30):
    ichimoku_df, _ = ta.ichimoku(
        df["high"], df["low"], df["close"], tenkan=tenkan, kijun=kijun, senkou=senkou
    )
    df = pd.concat([df, ichimoku_df], axis=1)
    return df


def add_rvi(df):
    common_params = {
        "scalar": 100,
        "refined": False,
        "thirds": False,
        "mamode": "ema",
        "drift": None,
        "offset": None,
    }

    rvi_15 = ta.rvi(df["close"], df["high"], df["low"], length=15, **common_params)
    df["RVI_15"] = rvi_15

    rvi_10 = ta.rvi(df["close"], df["high"], df["low"], length=10, **common_params)
    df["RVI_10"] = rvi_10

    rvi_5 = ta.rvi(df["close"], df["high"], df["low"], length=5, **common_params)
    df["RVI_5"] = rvi_5

    return df


def add_trix(df):
    trix_df0 = ta.trix(df["close"], length=18, signal=9)
    trix_df0.columns = ["TRIX_18_9", "TRIXs_18_9"]

    trix_df1 = ta.trix(df["close"], length=12, signal=6)
    trix_df1.columns = ["TRIX_12_6", "TRIXs_12_6"]

    trix_df2 = ta.trix(df["close"], length=10, signal=5)
    trix_df2.columns = ["TRIX_10_5", "TRIXs_10_5"]

    df = pd.concat([df, trix_df0, trix_df1, trix_df2], axis=1)
    return df


def add_mfi(df):
    df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"])
    return df


def add_macd_columns_6_13_5(df):
    macd = ta.macd(df["close"], fast=6, slow=13, signal=5)
    df = pd.concat([df, macd.add_suffix("_6_13_5")], axis=1)
    return df


def add_stochastic_oscillator_7_3_3(df):
    stoch = ta.stoch(df["high"], df["low"], df["close"], fast_k=7, slow_k=3, slow_d=3)
    df = pd.concat([df, stoch.add_suffix("_7_3_3")], axis=1)
    return df


def add_stochastic_oscillator_10_3_3(df):
    stoch = ta.stoch(df["high"], df["low"], df["close"], fast_k=10, slow_k=3, slow_d=3)
    df = pd.concat([df, stoch.add_suffix("_10_3_3")], axis=1)
    return df


def add_rsi(df):
    for length in [5, 10, 14]:
        df[f"RSI_{length}"] = ta.rsi(df["close"], length=length)
    return df


# Directory containing the CSV files
csv_directory = "data/kc/btc/heiken_ashi/raw"

# Directory to save the updated CSV files
output_directory = "data/kc/btc/heiken_ashi/with_trade_indicators/raw"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through each file in the directory
for file_name in os.listdir(csv_directory):
    if file_name.endswith(".csv"):
        file_path = os.path.join(csv_directory, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Add the following columns
        df = add_standard_pivot_points(df)
        df = add_rsi(df)
        df = add_atr(df)
        df = add_roc(df)
        df = add_cci(df)
        df = add_cmf(df)
        df = add_obv(df)
        df = add_mfi(df)
        df = add_rvi(df)
        df = add_psar(df)
        df = add_trix(df)
        df = add_sma(df)
        df = add_ema(df)
        df = add_ichimoku(df)
        df = add_bollinger_bands(df)
        df = add_bollinger_bandwidth(df)
        df = add_macd_columns(df)
        df = add_macd_columns_6_13_5(df)
        df = add_stochastic_oscillator(df)
        df = add_stochastic_oscillator_7_3_3(df)
        df = add_stochastic_oscillator_10_3_3(df)

        # Create a new file name with '_ti' appended before the file extension
        new_file_name = file_name[:-4] + "_ti.csv"
        new_file_path = os.path.join(output_directory, new_file_name)

        # Save the updated DataFrame to the new file path
        df.to_csv(new_file_path, index=False)
        print(
            f"Updated {file_name} with trade indicator columns and saved as {new_file_name} in {output_directory}"
        )
