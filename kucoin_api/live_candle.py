from kucoin.ws_client import KucoinWsClient


def handle_kline(socket_data):
    print("Kline data:", socket_data)


socket = KucoinWsClient(
    on_msg=handle_kline, on_error=print, on_close=print, on_open=print
)
socket.start_kline(symbol="BTC-USDT", granularity=60)
