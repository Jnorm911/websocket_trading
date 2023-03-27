import time
import pandas as pd
from datetime import datetime, timedelta
from kucoin.client import Market
import time
import random


def fetch_candles(start_time, end_time, granularity, symbol):
    client = Market(url="https://api.kucoin.com")
    klines = client.get_kline(
        symbol,
        granularity,
        startAt=int(start_time.timestamp()),
        endAt=int(end_time.timestamp()),
    )
    return klines


def fetch_candles_with_retry(
    start_time, end_time, granularity, symbol, max_retries=5, delay=1
):
    retries = 0

    while retries < max_retries:
        try:
            return fetch_candles(start_time, end_time, granularity, symbol)
        except Exception as e:
            if "429" in str(e):
                print("Rate limit error. Retrying after a delay...")
                time.sleep(delay * (2**retries) + random.uniform(0.1, 1))
                retries += 1
            else:
                raise e
    raise Exception(f"Failed to fetch candles after {max_retries} retries.")


# Symbol for Bitcoin
symbol = "BTC-USDT"

# Define the start and end times for 3 months of data
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=90)

# Set the granularity for 1-minute candles
granularity = "1min"


# Initialize an empty list to store data
data = []

# Fetch data in 1-day intervals
interval = timedelta(days=1)
current_start = start_time

while current_start < end_time:
    current_end = current_start + interval
    data.extend(
        fetch_candles_with_retry(current_start, current_end, granularity, symbol)
    )
    current_start = current_end

# Create a pandas DataFrame from the response data
df = pd.DataFrame(
    data, columns=["start_time", "open", "close", "high", "low", "volume", "turnover"]
)

# Calculate the average trading volume based on the previous 100 candles
df["avg_volume_100"] = df["volume"].rolling(window=100).mean()

# Add a "color" column based on the closing price comparison
df["color"] = "grey"  # Initialize all rows with "grey"
df.loc[df["close"] > df["close"].shift(1), "color"] = "green"
df.loc[df["close"] < df["close"].shift(1), "color"] = "red"

# Convert the start_time column to datetime format and set timezone to America/Los_Angeles
df["start_time"] = pd.to_datetime(df["start_time"], unit="s")
df["start_time"] = (
    df["start_time"]
    .dt.tz_localize("UTC")
    .dt.tz_convert("America/Los_Angeles")
    .dt.tz_localize(None)
)

# Save the DataFrame to a CSV file
file_path = "data/kucoin/BTC/1min.csv"
df.to_csv(file_path, index=False)
