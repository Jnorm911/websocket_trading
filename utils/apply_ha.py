import pandas as pd
import os


def heiken_ashi(df):
    ha_df = pd.DataFrame(index=df.index, columns=df.columns)
    ha_df["time"] = df["time"]
    ha_df["open"] = (df["open"] + df["close"]) / 2
    ha_df["close"] = (df["open"] + df["close"] + df["high"] + df["low"]) / 4
    ha_df["high"] = df[["high", "open", "close"]].max(axis=1)
    ha_df["low"] = df[["low", "open", "close"]].min(axis=1)
    ha_df["volume"] = df["volume"]
    ha_df["turnover"] = df["turnover"]
    ha_df["color"] = df["color"]
    ha_df["avg_vol_last_100"] = df["avg_vol_last_100"]
    return ha_df


def save_data_to_csv(data, minute_length, suffix):
    filename = f"kc_btc_{minute_length}min{suffix}.csv"
    data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


for minute_length in range(1, 61):
    # Read the original CSV file
    input_filename = f"kc_btc_{minute_length}min.csv"
    if os.path.exists(input_filename):
        data = pd.read_csv(input_filename)

        # Convert the 'time' column to datetime
        data["time"] = pd.to_datetime(data["time"])

        # Calculate Heikin-Ashi candles
        ha_data = heiken_ashi(data)

        # Save the data to a CSV file with '_ha' appended to the name
        save_data_to_csv(ha_data, minute_length, "_ha")
    else:
        print(f"File {input_filename} not found.")
