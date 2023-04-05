import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

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

    essential_columns = [
        "volume",
        "turnover",
        "avg_vol_last_100",
        "MACD_12_26_9",
        "MACDh_12_26_9",
        "MACDs_12_26_9",
        "RSI",
        "ROC",
        "STOCHk_14_3_3",
        "STOCHd_14_3_3",
        "ATR",
        "BBL_5_2.0",
        "BBU_5_2.0",
        "BBM_5_2.0",
        "BBP_5_2.0",
        "BBB_5_2.0",
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "EMA_5",
        "EMA_10",
        "EMA_20",
        "CCI",
        "PP",
        "R1",
        "R2",
        "R3",
        "S1",
        "S2",
        "S3",
    ]

    # standard
    standard_scaler = StandardScaler()
    df[essential_columns] = standard_scaler.fit_transform(df[essential_columns])

    # minmax
    # minmax_scaler = MinMaxScaler()
    # df[essential_columns] = minmax_scaler.fit_transform(df[essential_columns])

    # robust
    # robust_scaler = RobustScaler()
    # df[essential_columns] = robust_scaler.fit_transform(df[essential_columns])

    # Save the preprocessed data to a new CSV file
    df.to_csv(preprocessed_data_path, index=False)

    print(
        f"Preprocessing complete for {i}min. Preprocessed data saved to: {preprocessed_data_path}"
    )

print("All files preprocessed and saved.")


for i in range(1, 61):
    # Define the input file path
    preprocessed_data_path = f"data/kc/btc/heiken_ashi/with_trade_indicators/processed/kc_btc_{i}min_ha_ti_pro.csv"

    # Read the preprocessed CSV file
    df = pd.read_csv(preprocessed_data_path)

    # Define your target variable (column) and remove it from the features list
    target_column = "color_change"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a validation set (further split the training set)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )

    print(
        f"Data for {i}min has been split into training, validation, and testing sets."
    )
