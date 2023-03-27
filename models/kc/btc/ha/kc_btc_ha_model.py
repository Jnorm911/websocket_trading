import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def create_trend_feature(df, window_size):
    df[f"trend_{window_size}"] = (
        (df["color"] != df["color"].shift(1)).rolling(window=window_size).sum()
    )


def create_signal(df):
    # In this example, we'll consider the trend change from the previous candle as the signal
    return (df["color"] != df["color"].shift(1)).astype(int)


def preprocess_data(df, window_sizes):
    for window in window_sizes:
        create_trend_feature(df, window)

    # Create the signal based on the trend change
    df["signal"] = create_signal(df)

    # Drop unnecessary columns
    df.drop(columns=["time", "color"], inplace=True)

    return df


def train_and_evaluate_model(df):
    # Prepare data
    X = df.drop(columns=["signal"])
    y = df["signal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return clf, accuracy, report


# Main loop to preprocess, train, and evaluate models for all timeframes
models = {}
window_sizes = [5, 10, 15]  # Add more window sizes if needed
for i in range(1, 61):
    filename = f"kc_btc_{i}min_ha.csv"
    df = pd.read_csv(filename)
    df = preprocess_data(df, window_sizes)
    model, accuracy, report = train_and_evaluate_model(df)

    print(f"Model for {i}min candle:")
    print(f"Accuracy: {accuracy}")
    print(report)
    print("\n")

    models[i] = model
