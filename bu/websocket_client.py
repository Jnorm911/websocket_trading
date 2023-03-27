import pytz
import time
import asyncio
import pandas as pd
from datetime import datetime
from kucoin.client import WsToken
from kucoin.ws_client import KucoinWsClient
from websockets.exceptions import ConnectionClosedError
from strategy.heikin_ashi import calculate_heikin_ashi

symbol = "BTC-USDT"
desired_timezone = pytz.timezone("America/Los_Angeles")

start_time = int(time.time())


previous_timestamp = 0


async def handle_message(msg, dataframe):
    global previous_timestamp
    if msg["type"] == "message":
        candles = msg["data"]["candles"]
        timestamp = int(candles[0])

        if timestamp != previous_timestamp:
            ha_open, ha_close, ha_high, ha_low = calculate_heikin_ashi(candles)
            volume = float(candles[5])

            # Convert the timestamp to your desired timezone
            utc_time = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.UTC)
            local_time = utc_time.astimezone(desired_timezone)

            # Set the color based on the current Heikin Ashi close price compared to the previous Heikin Ashi close price
            if dataframe.empty:
                color = "gray"
            else:
                prev_ha_close = dataframe.iloc[-1]["ha_close"]
                if prev_ha_close < ha_close:
                    color = "green"
                elif prev_ha_close > ha_close:
                    color = "red"
                else:
                    color = dataframe.iloc[-1]["color"]

            row = {
                "start_time": local_time,
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

    return dataframe


async def print_dataframe(dataframe):
    dataframe["average_volume"] = dataframe["volume"].sum() / len(dataframe)
    print(dataframe)  # Print DataFrame to console
    dataframe.to_csv(
        "C:\\Users\\jnorm\\Projects\\dataframe.csv", index=False
    )  # Save DataFrame to CSV file
    return dataframe


async def main():
    global dataframe
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

    while True:
        try:
            ws_token_client = WsToken()
            ws_client = await KucoinWsClient.create(
                None, ws_token_client, None, private=False
            )

            await ws_client.subscribe(f"/market/candles:{symbol}_1min")

            while True:
                msg = await ws_client.recv()
                dataframe = await handle_message(msg, dataframe)
                dataframe = await print_dataframe(dataframe)
                await asyncio.sleep(60)
        except ConnectionClosedError as e:
            print(f"Connection closed: {e}")
            # Add custom reconnection logic or error handling here
            await asyncio.sleep(5)  # Wait for 5 seconds before trying to reconnect


if __name__ == "__main__":
    asyncio.run(main())
