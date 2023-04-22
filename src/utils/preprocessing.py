import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def preprocess_data(df, columns_to_scale, scaler_type):
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaler type")

    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df


scaler_types = ["standard", "minmax", "robust"]

for scaler_type in scaler_types:
    for i in range(1, 61):
        data_path = (
            f"data/kc/btc/heiken_ashi/with_trade_indicators/raw/kc_btc_{i}min_ha_ti.csv"
        )
        preprocessed_data_path = f"data/kc/btc/heiken_ashi/with_trade_indicators/{scaler_type}/kline/kc_btc_{i}min_ha_ti_pro.csv"

        df = pd.read_csv(data_path)

        print(f"Missing values for {i}min:")
        print(df.isnull().sum())

        df = df.dropna()
        df = df[(df["color"] != "grey") & (df["color"] != "gray")]

        df["color_binary"] = (df["color"] == "green").astype(int)
        df["color_shifted"] = df["color_binary"].shift(1)
        df["color_change"] = (df["color_binary"] != df["color_shifted"]).astype(int)

        df.drop(columns=["color_binary", "color_shifted", "color"], inplace=True)

        # Define the columns to scale
        columns_to_scale = [
            "open",
            "close",
            "high",
            "low",
            "volume",
            "turnover",
            "avg_vol_last_100",
            # "RSI",
            # "MACD_12_26_9",
            # "MACDh_12_26_9",
            # "MACDs_12_26_9",
            # "CCI",
            # "ATR",
            # "ROC",
        ]

        df = preprocess_data(df, columns_to_scale, scaler_type)

        df.to_csv(preprocessed_data_path, index=False)

        print(
            f"Preprocessing complete for {i}min with {scaler_type} scaler. Preprocessed data saved to: {preprocessed_data_path}"
        )

    print(f"All files preprocessed and saved for {scaler_type} scaler.")
