import asyncio
import pandas as pd
from kucoin.client import WsToken
from kucoin.ws_client import KucoinWsClient
import candle_manager

symbol = "BTC-USDT"
candle_duration = "1min"

current_candles = [None] * 60
ha_historical_candles = [
    pd.DataFrame(
        columns=[
            "start_time",
            "duration",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "avg_volume",
            "ha_open",
            "ha_close",
            "ha_high",
            "ha_low",
            "color",
        ]
    )
    for _ in range(60)
]


async def handle_message(msg):
    if msg["topic"] == f"/market/candles:{symbol}_{candle_duration}":
        candle_data = msg["data"]["candles"]
        trade = {
            "timestamp": int(candle_data[0]),
            "price": float(candle_data[2]),
            "amount": float(candle_data[5]),
        }
        global current_candles, ha_historical_candles
        current_candles, ha_historical_candles = candle_manager.process_trade(
            trade, current_candles, ha_historical_candles
        )

        for duration in range(1, 61):
            if current_candles[duration - 1] is not None:
                current_df = pd.DataFrame([current_candles[duration - 1]])
                if not current_df.empty:
                    try:
                        ha_candle = candle_manager.create_heikin_ashi_candles(
                            current_df
                        )
                        ha_historical_candles[duration - 1] = ha_historical_candles[
                            duration - 1
                        ].append(ha_candle, ignore_index=True)
                    except KeyError as e:
                        print(f"KeyError encountered: {e}")


async def main():
    ws_token_client = WsToken()
    ws_client = await KucoinWsClient.create(
        None, ws_token_client, handle_message, private=False
    )

    # Subscribe to the Kline channel
    await ws_client.subscribe(f"/market/candles:{symbol}_{candle_duration}")

    while True:
        await asyncio.sleep(0.01)  # Update every 15 seconds


if __name__ == "__main__":
    asyncio.run(main())
