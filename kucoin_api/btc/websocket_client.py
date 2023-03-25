import pytz
import time
import asyncio
import pandas as pd
from datetime import datetime
from kucoin.client import WsToken
from kucoin.ws_client import KucoinWsClient
from strategy.heikin_ashi import calculate_heikin_ashi

symbol = "BTC-USDT"
desired_timezone = pytz.timezone("America/Los_Angeles")

dataframe = pd.DataFrame(
    columns=[
        "start_time",
        "duration",
        "ha_open",
        "ha_close",
        "ha_high",
        "ha_low",
        "volume",
        "color",
        "average_volume",
    ]
)

start_time = int(time.time())


previous_timestamp = 0
dataframe = pd.DataFrame(
    columns=[
        "start_time",
        "duration",
        "ha_open",
        "ha_close",
        "ha_high",
        "ha_low",
        "volume",
        "color",
        "average_volume",
    ]
)


async def handle_message(msg):
    global dataframe, previous_timestamp
    if msg["type"] == "message":
        candles = msg["data"]["candles"]
        timestamp = int(candles[0])

        if timestamp != previous_timestamp:
            ha_open, ha_close, ha_high, ha_low = calculate_heikin_ashi(candles)
            volume = float(candles[5])

            # Convert the timestamp to your desired timezone
            utc_time = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.UTC)
            local_time = utc_time.astimezone(desired_timezone)

            # Set the color to grey for the first row, otherwise use the calculated color
            color = (
                "grey" if dataframe.empty else "green" if ha_close > ha_open else "red"
            )

            row = {
                "start_time": local_time,  # Use local_time instead of timestamp
                "duration": "1min",
                "ha_open": ha_open,
                "ha_close": ha_close,
                "ha_high": ha_high,
                "ha_low": ha_low,
                "volume": volume,
                "color": color,
                "average_volume": 0,  # Placeholder for now
            }
            dataframe = pd.concat([dataframe, pd.DataFrame([row])], ignore_index=True)

            # Remove the oldest row if the dataframe has more than 100 rows
            if len(dataframe) > 100:
                dataframe = dataframe.iloc[1:].reset_index(drop=True)

            previous_timestamp = timestamp


async def print_dataframe():
    global dataframe
    while True:
        await asyncio.sleep(60)
        dataframe["average_volume"] = dataframe["volume"].sum() / len(dataframe)
        print("Last 100 rows of the 1-minute dataframe:")
        print(dataframe.tail(100))


async def main():
    ws_token_client = WsToken()
    ws_client = await KucoinWsClient.create(
        None, ws_token_client, handle_message, private=False
    )

    await ws_client.subscribe(f"/market/candles:{symbol}_1min")

    asyncio.create_task(print_dataframe())  # Schedule the printing function

    while True:
        await asyncio.sleep(15)


if __name__ == "__main__":
    asyncio.run(main())
