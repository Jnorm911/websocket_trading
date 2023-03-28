import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report


def create_trend_feature(df, window_size):
    df[f"trend_{window_size}"] = (
        (df["color"] != df["color"].shift(1)).rolling(window=window_size).sum()
    )


def moving_average(df, window):
    return df["close"].rolling(window=window).mean()


def exponential_moving_average(df, window):
    return df["close"].ewm(span=window).mean()


def momentum(df, window):
    return df["close"] - df["close"].shift(window)


def rate_of_change(df, window):
    return (df["close"] - df["close"].shift(window)) / df["close"].shift(window)


def relative_strength_index(df, window):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def create_features(df):
    df["moving_average_5"] = moving_average(df, 5)
    df["moving_average_10"] = moving_average(df, 10)
    df["moving_average_20"] = moving_average(df, 20)
    df["ema_5"] = exponential_moving_average(df, 5)
    df["ema_10"] = exponential_moving_average(df, 10)
    df["ema_20"] = exponential_moving_average(df, 20)
    df["momentum_5"] = momentum(df, 5)
    df["momentum_10"] = momentum(df, 10)
    df["roc_5"] = rate_of_change(df, 5)
    df["roc_10"] = rate_of_change(df, 10)
    df["rsi_14"] = relative_strength_index(df, 14)


def create_signal(df):
    # In this example, we'll consider the trend change from the previous candle as the signal
    return (df["color"] != df["color"].shift(1)).astype(int)


def preprocess_data(df, window_sizes):
    for window in window_sizes:
        create_trend_feature(df, window)

    df.loc[df["color"] == "grey", "color"] = df["color"].shift(1)

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

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return clf, accuracy, report


def predict_with_trained_models(models, current_df, window_sizes):
    # Preprocess the current_df
    current_df = preprocess_data(current_df, window_sizes)

    # Prepare data
    X_current = current_df.drop(columns=["signal"])
    y_current = current_df["signal"]

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_current_imputed = imputer.fit_transform(X_current)

    # Scale data
    scaler = StandardScaler()
    X_current_scaled = scaler.fit_transform(X_current_imputed)

    # Use trained models to make predictions
    predictions = {}
    for i, model in models.items():
        y_pred = model.predict(X_current_scaled)
        accuracy = accuracy_score(y_current, y_pred)
        report = classification_report(y_current, y_pred)

        print(f"Model for {i}min candle:")
        print(f"Accuracy: {accuracy}")
        print(report)
        print("\n")

        predictions[i] = {
            "predictions": y_pred,
            "accuracy": accuracy,
            "classification_report": report,
        }

    return predictions


models = {}
window_sizes = [5, 10, 15, 20]  # Add more window sizes if needed
data_dir = "data/kc/btc/heiken_ashi"

# Initialize an empty list to store evaluation results
evaluation_results = []

# Main loop to preprocess, train, and evaluate models for all timeframes
for i in range(1, 61):
    filename = f"kc_btc_{i}min_ha.csv"
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path)
    df = preprocess_data(df, window_sizes)
    model, accuracy, report = train_and_evaluate_model(df)

    # Extract F1-score for the positive class (1) from the classification report
    f1_score = float(report.split("\n")[2].split()[3])

    # Append evaluation results to the list
    evaluation_results.append({"Candle": i, "Accuracy": accuracy, "F1-score": f1_score})

    # Store the trained model
    models[i] = model


# Read the existing evaluation results
output_file_path = "evaluation_results.csv"


def read_existing_results(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame()


existing_results = read_existing_results(output_file_path)


def get_next_version_number(existing_results):
    if not existing_results.empty:
        max_version = existing_results["Test"].apply(lambda x: int(x[1:])).max()
        return f"v{max_version + 1}"
    else:
        return "v1"


next_version = get_next_version_number(existing_results)

existing_results = read_existing_results(output_file_path)

# Determine the next version number
next_version = (
    f"v{existing_results['Test'].max() + 1}" if not existing_results.empty else "v1"
)

# Main loop to preprocess, train, and evaluate models for all timeframes
for i in range(1, 61):
    filename = f"kc_btc_{i}min_ha.csv"
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path)
    df = preprocess_data(df, window_sizes)
    model, accuracy, report = train_and_evaluate_model(df)

    # Extract F1-score for the positive class (1) from the classification report
    f1_score = float(report.split("\n")[2].split()[3])

    # Append evaluation results to the list
    evaluation_results.append({"Candle": i, "Accuracy": accuracy, "F1-score": f1_score})

    # Store the trained model
    models[i] = model

# Convert the list of evaluation results to a DataFrame
evaluation_df = pd.DataFrame(evaluation_results)

# Add the Test column to the sorted evaluation results
evaluation_df["Test"] = next_version

# Append the new results to the existing results
updated_results = pd.concat([existing_results, evaluation_df], ignore_index=True)


# Save the updated evaluation results to the same output file
def save_updated_results(file_path, df):
    df.to_csv(file_path, index=False)


save_updated_results(output_file_path, updated_results)

# Display the updated evaluation results
print(updated_results)


# # Load the current file for which you want to make predictions
# current_file_path = "path/to/your/current_file.csv"
# current_df = pd.read_csv(current_file_path)

# # Make predictions using the trained models
# predictions = predict_with_trained_models(models, current_df, window_sizes)

# # Print the predictions for each model
# for i, pred in predictions.items():
#     print(f"Model for {i}min candle:")
#     print(f"Predictions: {pred['predictions']}")
#     print(f"Accuracy: {pred['accuracy']}")
#     print(pred["classification_report"])
#     print("\n")
