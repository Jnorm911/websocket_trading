import os

import pandas as pd
from sklearnex import patch_sklearn
from tqdm.auto import tqdm

patch_sklearn()
import numpy as np
from boruta import BorutaPy
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

rf_model = RandomForestClassifier()


def apply_feature_selection(X, y, estimator):
    model = rf_model

    # Recursive Feature Elimination with Cross-Validation (RFECV)
    rfecv_selector, rfecv_features, _ = custom_rfecv_feature_selection(X, y, model)

    # Feature Importance from RandomForestClassifier
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = X.columns[indices].tolist()

    # Boruta Feature Selection
    boruta_selector, boruta_features = custom_boruta_feature_selection(X, y, model)

    selected_features = {
        "RFECV": rfecv_features,
        "Feature Importance": {n: sorted_features[:n] for n in range(6, 25)},
        "Boruta": boruta_features,
    }

    return selected_features


def fit_and_evaluate(X, y, selected_features):
    model = rf_model
    results = {}
    tscv = TimeSeriesSplit(n_splits=5)

    for method, features in selected_features.items():
        if method == "Feature Importance":
            best_score = 0
            best_features = None
            for n in range(6, 25):
                current_features = features[n]
                scores = []
                for train_index, test_index in tscv.split(X):
                    X_train, X_test = (
                        X[current_features].iloc[train_index],
                        X[current_features].iloc[test_index],
                    )
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                    scores.append(score)

                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_features = current_features

            results[method] = {"best_features": best_features, "accuracy": best_score}
        else:
            # Handle other methods (RFECV and Boruta)
            scores = []
            for train_index, test_index in tscv.split(X):
                X_train, X_test = (
                    X[features].iloc[train_index],
                    X[features].iloc[test_index],
                )
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)

            results[method] = {"best_features": features, "accuracy": np.mean(scores)}

    return results


# functions with custom_boruta_feature_selection function
def custom_boruta_feature_selection(X_train, y_train, estimator):
    # Update the RandomForestClassifier with a higher number of estimators
    estimator.set_params(n_estimators=500)

    # Create Boruta selector with the updated parameters
    boruta_selector = BorutaPy(
        estimator,
        n_estimators="auto",
        verbose=0,
        random_state=42,
        max_iter=200,
        alpha=0.1,
    )
    boruta_selector.fit(np.array(X_train), np.array(y_train))
    selected_features = X_train.columns[boruta_selector.support_].tolist()

    return boruta_selector, selected_features


def custom_rfecv_feature_selection(X_train, y_train, estimator):
    rfecv_selector = RFECV(estimator, step=1, cv=5)
    rfecv_selector.fit(X_train, y_train)
    selected_features = X_train.columns[rfecv_selector.get_support()].tolist()

    return rfecv_selector, selected_features, rfecv_selector.n_features_


def preprocess_data(data):
    features = data.drop(columns=["time", "color_change"])

    # Normalize all numerical features using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    data[features.columns] = scaled_features

    return data


def process_file(i):
    FOLDER_PATH = os.path.join(
        "data",
        "kc",
        "btc",
        "heiken_ashi",
        "with_trade_indicators",
        "minmax",
        "kline",
    )
    file_path = os.path.join(FOLDER_PATH, f"kc_btc_{i}min_ha_ti_pro.csv")
    data = pd.read_csv(file_path)

    # Preprocess the data
    data = preprocess_data(data)

    target = "color_change"
    X = data.drop(columns=[target])
    y = data[target]

    # Check the class distribution
    class_counts = y.value_counts()
    total_samples = len(y)
    class_distribution = class_counts / total_samples

    print(f"Class distribution for {i}min dataset:\n", class_distribution)

    # Apply feature selection
    selected_features = apply_feature_selection(X, y, rf_model)

    # Fit models and evaluate
    results = fit_and_evaluate(X, y, selected_features)

    # Update the all_results DataFrame
    new_results = []
    for method, result in results.items():
        new_results.append(
            {
                "model": f"Random Forest with {method} Feature Selection",
                "duration": i,
                "best_features": result["best_features"],
                "accuracy": result["accuracy"],
            }
        )

    return pd.DataFrame(new_results)


if __name__ == "__main__":
    # Loop through each file, apply feature selection, fit models, and store the results
    all_results_list = Parallel(n_jobs=-1)(
        delayed(process_file)(i) for i in tqdm(range(3, 61, 3), desc="Processing files")
    )

    # Combine the results from all parallel tasks
    all_results = pd.concat(all_results_list, ignore_index=True)

    # Save the final results DataFrame to a CSV file
    if not all_results.empty:
        all_results.to_csv("rf_minmax.csv", index=False)
        print("All results saved to 'rf_minmax.csv'")
    else:
        print("No results to save.")
