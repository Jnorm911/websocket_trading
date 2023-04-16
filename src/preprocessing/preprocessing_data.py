import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

for i in range(1, 61):
    # Define the input and output file paths
    data_path = f"data/kc/btc/heiken_ashi/with_trade_indicators/kc_btc_{i}min_ha_ti.csv"
    preprocessed_data_path = f"data/kc/btc/heiken_ashi/with_trade_indicators/processed/kc_btc_{i}min_ha_ti_pro.csv"

    # Read the input CSV file
    df = pd.read_csv(data_path)

    # Check for missing values
    print(f"Missing values for {i}min:")
    print(df.isnull().sum())

    # Preprocess the data
    df = df.dropna()
    df = df[(df["color"] != "grey") & (df["color"] != "gray")]

    # Create the color_change column
    df["color_binary"] = (df["color"] == "green").astype(int)
    df["color_shifted"] = df["color_binary"].shift(1)
    df["color_change"] = (df["color_binary"] != df["color_shifted"]).astype(int)

    # Drop the unnecessary columns
    df.drop(columns=["color_binary", "color_shifted", "color"], inplace=True)

    numeric_columns = df.select_dtypes(include=["number"]).columns

    # Exclude the "time" and "color_change" columns
    columns_to_scale = numeric_columns.drop(["time", "color_change"])

    # Standard scaler
    standard_scaler = StandardScaler()
    df[columns_to_scale] = standard_scaler.fit_transform(df[columns_to_scale])

    # minmax
    # minmax_scaler = MinMaxScaler()
    # df[columns_to_scale] = minmax_scaler.fit_transform(df[columns_to_scale])

    # robust
    # robust_scaler = RobustScaler()
    # df[columns_to_scale] = robust_scaler.fit_transform(df[columns_to_scale])

    # Save the preprocessed data to a new CSV file
    df.to_csv(preprocessed_data_path, index=False)

    print(
        f"Preprocessing complete for {i}min. Preprocessed data saved to: {preprocessed_data_path}"
    )

print("All files preprocessed and saved.")
