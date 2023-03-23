def apply_heiken_ashi(data):
    heiken_ashi_data = data.copy()

    heiken_ashi_data["HA_Close"] = (
        data["open"] + data["high"] + data["low"] + data["close"]
    ) / 4
    heiken_ashi_data["HA_Open"] = (data["open"].shift(1) + data["close"].shift(1)) / 2
    heiken_ashi_data["HA_High"] = data[["high", "HA_Open", "HA_Close"]].max(axis=1)
    heiken_ashi_data["HA_Low"] = data[["low", "HA_Open", "HA_Close"]].min(axis=1)

    return heiken_ashi_data
