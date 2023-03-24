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


def process_trade(trade, current_candles, historical_candles):
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
            avg_volume = historical_candles[duration - 1]["volume"].mean()

            # Add the average trading volume to the current candle
            current_candles[duration - 1]["avg_volume"] = avg_volume

            print(
                f"Number of rows in historical_candles for {duration}-minute candles: {len(historical_candles[duration - 1])}"
            )

            # Use pd.concat to add the current_candles[duration - 1] to historical_candles[duration - 1]
            historical_candles[duration - 1] = pd.concat(
                [
                    historical_candles[duration - 1],
                    pd.DataFrame([current_candles[duration - 1]]),
                ],
                ignore_index=True,
            )

            historical_candles[duration - 1] = historical_candles[duration - 1].tail(
                100
            )

            print(f"{duration}-minute candle:", current_candles[duration - 1])

            current_candles[duration - 1] = create_new_candle(
                timestamp, price, duration
            )

        update_candle(current_candles[duration - 1], price, amount)

    return current_candles, historical_candles
