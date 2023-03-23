from kucoin_api.rest import get_historical_klines
from datetime import datetime, timedelta

symbol = "BTC-USDT"
interval = 1  # 1-minute candles (supported interval)

# Calculate the current time in UTC
current_time = datetime.utcnow()

# Calculate the end_time as the start of the current hour
end_time = current_time.replace(minute=0, second=0, microsecond=0)

# Calculate the start_time as 1 hour before the end_time
start_time = end_time - timedelta(hours=1)

# Ensure start_time starts at the 00 hour
start_time = start_time.replace(minute=0, second=0, microsecond=0)

# Subtract any additional hours from start_time to ensure it starts at the 00 hour
hours_to_subtract = start_time.hour % 24
start_time = start_time - timedelta(hours=hours_to_subtract)

print(start_time)
print(end_time)
# Fetch data for the last hour
historical_data = get_historical_klines(
    symbol,
    interval,
    start_time.strftime("%Y-%m-%dT%H:%M:%S"),
    end_time.strftime("%Y-%m-%dT%H:%M:%S"),
)

print(historical_data.head())
