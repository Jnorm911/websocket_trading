def calculate_heikin_ashi(candles):
    open_price = float(candles[1])
    close_price = float(candles[2])
    high_price = float(candles[3])
    low_price = float(candles[4])

    ha_open = (open_price + close_price) / 2
    ha_close = (open_price + high_price + low_price + close_price) / 4
    ha_high = max(high_price, ha_open, ha_close)
    ha_low = min(low_price, ha_open, ha_close)

    return ha_open, ha_close, ha_high, ha_low
