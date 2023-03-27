import time
import pandas as pd
from kucoin.client import Market

day_in_seconds = 60 * 60 * 24


def fetch_kucoin_data(symbol, kline_type, start_at, end_at):
    market = Market()
    data = market.get_kline(symbol, kline_type, startAt=start_at, endAt=end_at)
    return data


def assign_color(row):
    if row["open"] == row["close"]:
        return "grey"
    elif row["open"] > row["close"]:
        return "red"
    else:
        return "green"


def save_data_to_csv(data, day):
    filename = f"kc_btc_1min_day_{day}.csv"
    data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


symbol = "BTC-USDT"
kline_type = "1min"
end_at = int(time.time())
start_at = end_at - 60 * 60 * 24 * 7 * 12

# Set the starting day
current_day = 0

# Calculate the total number of days
num_days = (end_at - start_at) // day_in_seconds

while current_day <= num_days:
    day_start_at = start_at + current_day * day_in_seconds
    day_end_at = min(day_start_at + day_in_seconds, end_at)

    data = fetch_kucoin_data(symbol, kline_type, day_start_at, day_end_at)

    if data:
        new_data = pd.DataFrame(
            data, columns=["time", "open", "close", "high", "low", "volume", "turnover"]
        )
        new_data["time"] = pd.to_datetime(new_data["time"], unit="s")
        new_data["color"] = new_data.apply(assign_color, axis=1)

        # Compute the rolling average volume for the last 100 rows
        new_data["avg_vol_last_100"] = (
            new_data["volume"].rolling(window=100, min_periods=1).mean()
        )

        save_data_to_csv(new_data, current_day)
        current_day += 1
    else:
        print(f"Error fetching data for day {current_day}. Retrying...")

    time.sleep(1)  # To avoid reaching API rate limits
