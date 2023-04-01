import pandas as pd

## WARNING ##
## TODO ##
# Grey should only ever exist on the first row, time should be date and time in 2 different columns.


def assign_color(row):
    if row["open"] == row["close"]:
        return "grey"
    elif row["open"] > row["close"]:
        return "red"
    else:
        return "green"


def save_data_to_csv(data, minute_length):
    filename = f"kc_btc_{minute_length}min.csv"
    data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# Read the combined CSV file
combined_data = pd.read_csv("kc_btc_1min_raw.csv")

# Convert the 'time' column to datetime
combined_data["time"] = pd.to_datetime(combined_data["time"])

# Set the 'time' column as the index
combined_data = combined_data.set_index("time")

for minute_length in range(1, 61):
    # Resample the 1-minute candles
    resampled_data = combined_data.resample(f"{minute_length}T").agg(
        {
            "open": "first",
            "close": "last",
            "high": "max",
            "low": "min",
            "volume": "sum",
            "turnover": "sum",
        }
    )

    # Assign colors to the resampled candles
    resampled_data["color"] = resampled_data.apply(assign_color, axis=1)

    # Compute the rolling average volume for the last 100 rows
    resampled_data["avg_vol_last_100"] = (
        resampled_data["volume"].rolling(window=100, min_periods=1).mean()
    )

    # Check if the last row is an incomplete time window and drop it if necessary
    last_row_timestamp = resampled_data.index[-1]
    next_expected_timestamp = last_row_timestamp + pd.Timedelta(minutes=minute_length)
    if next_expected_timestamp > combined_data.index[-1] + pd.Timedelta(minutes=1):
        resampled_data = resampled_data.iloc[:-1]

    # Reset the index
    resampled_data = resampled_data.reset_index()

    # Save the data to a CSV file
    save_data_to_csv(resampled_data, minute_length)
