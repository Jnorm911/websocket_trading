import pandas as pd
from .my_client import market_client
from datetime import datetime


def get_historical_klines(symbol, interval, start_time, end_time=None):
    klines_type = f"{interval}hour"

    start_time_timestamp = int(
        datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S").timestamp()
    )
    end_time_timestamp = (
        int(datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S").timestamp())
        if end_time
        else None
    )

    klines = market_client.get_kline(
        symbol, klines_type, startAt=start_time_timestamp, endAt=end_time_timestamp
    )
    data = pd.DataFrame(
        klines,
        columns=["open_time", "open", "close", "high", "low", "volume", "turnover"],
    )

    # Convert Unix timestamp to datetime
    data["open_time"] = pd.to_datetime(data["open_time"], unit="s")

    return data
