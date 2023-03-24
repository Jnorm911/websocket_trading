import asyncio
from kucoin.client import WsToken
from kucoin.ws_client import KucoinWsClient
import pandas as pd
import candle_manager

symbol = "BTC-USDT"
candle_duration = "1min"

current_candles = [None] * 60
historical_candles = [
    pd.DataFrame(
        columns=["start_time", "open", "close", "high", "low", "volume", "duration"]
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
        global current_candles, historical_candles
        current_candles, historical_candles = candle_manager.process_trade(
            trade, current_candles, historical_candles
        )


async def main():
    ws_token_client = WsToken()
    ws_client = await KucoinWsClient.create(
        None, ws_token_client, handle_message, private=False
    )

    # Subscribe to the Kline channel
    await ws_client.subscribe(f"/market/candles:{symbol}_{candle_duration}")

    while True:
        await asyncio.sleep(15)  # Update every 15 seconds


if __name__ == "__main__":
    asyncio.run(main())
