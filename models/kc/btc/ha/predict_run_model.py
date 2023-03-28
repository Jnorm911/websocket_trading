import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_data(file_list):
    combined_df = pd.DataFrame()

    for file in file_list:
        df = pd.read_csv(file)
        df.loc[df["color"] == "grey", "color"] = df["color"].shift(1)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df


def create_features(df):
    # You can add more features based on domain knowledge or use feature engineering techniques
    df["moving_average_5"] = df["close"].rolling(window=5).mean()
    df["moving_average_10"] = df["close"].rolling(window=10).mean()


def create_target(df):
    df["consecutive_color_count"] = (
        (df["color"] != df["color"].shift(1)).groupby(df["color"]).cumsum()
    )


def prepare_data(file_list):
    df = preprocess_data(file_list)
    create_features(df)
    create_target(df)
    df.dropna(inplace=True)
    return df


def train_and_evaluate_model(df):
    # Prepare data
    X = df.drop(columns=["consecutive_color_count", "color"])
    y = df["consecutive_color_count"]
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


data_dir = "data/kc/btc/heiken_ashi"
file_list = [f"{data_dir}/kc_btc_{x}min_ha.csv" for x in range(1, 61)]

# Prepare the data
df = prepare_data(file_list)

# Train and evaluate the model
clf, accuracy, report = train_and_evaluate_model(df)

print("Accuracy:", accuracy)
print("Classification Report:", report)
