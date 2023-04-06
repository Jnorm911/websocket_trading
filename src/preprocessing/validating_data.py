import pandas as pd
from sklearn.model_selection import train_test_split

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
