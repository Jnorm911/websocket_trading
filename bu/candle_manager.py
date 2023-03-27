import pandas as pd


def create_new_candle(timestamp, price, duration):
    return {
        "start_time": timestamp,
        "open": price,
        "close": price,
        "high": price,
        "low": price,
        "volume": 0,
        "duration": duration,
    }


def update_candle(candle, price, amount):
    candle["close"] = price
    candle["high"] = max(candle["high"], price)
    candle["low"] = min(candle["low"], price)
    candle["volume"] += amount


def create_heikin_ashi_candles(candles):
    ha_candles = candles.copy()

    # Calculate Heikin Ashi Open
    if len(candles) > 1:
        ha_candles["ha_open"] = (
            candles["open"].shift(1) + candles["close"].shift(1)
        ) / 2
    else:
        ha_candles["ha_open"] = (candles["open"] + candles["close"]) / 2

    # Calculate Heikin Ashi Close
    ha_candles["ha_close"] = (
        candles["open"] + candles["high"] + candles["low"] + candles["close"]
    ) / 4

    # Calculate Heikin Ashi High and Low
    ha_candles["ha_high"] = candles[["high", "ha_open", "ha_close"]].max(axis=1)
    ha_candles["ha_low"] = candles[["low", "ha_open", "ha_close"]].min(axis=1)

    # Add color column
    ha_candles["color"] = "grey"
    ha_candles.loc[ha_candles["ha_close"] > ha_candles["ha_open"], "color"] = "green"
    ha_candles.loc[ha_candles["ha_close"] < ha_candles["ha_open"], "color"] = "red"

    return ha_candles


def process_trade(trade, current_candles, ha_historical_candles):
    timestamp = trade["timestamp"]
    price = trade["price"]
    amount = trade["amount"]

    for duration in range(1, 61):
        if current_candles[duration - 1] is None:
            current_candles[duration - 1] = create_new_candle(
                timestamp, price, duration
            )
        elif timestamp >= current_candles[duration - 1]["start_time"] + duration * 60:
            # Calculate the average trading volume
            avg_volume = ha_historical_candles[duration - 1]["volume"].mean()

            # Add the average trading volume to the current candle
            current_candles[duration - 1]["avg_volume"] = avg_volume

            current_candles[duration - 1] = create_new_candle(
                timestamp, price, duration
            )

        update_candle(current_candles[duration - 1], price, amount)

    return current_candles, ha_historical_candles
