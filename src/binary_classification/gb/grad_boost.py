import os

import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()
import shap
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from xgboost import XGBClassifier


def custom_rfecv_feature_selection(X_train, y_train, estimator):
    min_features_to_select = 6  # Minimum number of features to consider
    tscv = TimeSeriesSplit(n_splits=5)  # Time series cross-validation
    rfecv_selector = RFECV(
        estimator, step=1, cv=tscv, min_features_to_select=min_features_to_select
    )
    rfecv_selector.fit(X_train, y_train)

    # Get indices of features sorted by their rank
    indices = np.argsort(rfecv_selector.ranking_)

    selected_features_for_each_n = {}
    if rfecv_selector.n_features_ > 24:
        for n in range(6, 25):  # This will iterate from 6 to 24 inclusive
            selected_features_for_each_n[n] = X_train.columns[indices[:n]].tolist()
    else:
        selected_features_for_each_n[rfecv_selector.n_features_] = X_train.columns[
            indices[: rfecv_selector.n_features_]
        ].tolist()

    return rfecv_selector, selected_features_for_each_n, indices


def preprocess_data(data):
    features = data.drop(columns=["time", "color_change"])

    # Normalize all numerical features using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    data[features.columns] = scaled_features

    return data


def apply_feature_selection(X, y, estimator):
    model = estimator

    # Recursive Feature Elimination with Cross-Validation (RFECV)
    rfecv_selector, rfecv_features, _ = custom_rfecv_feature_selection(X, y, model)

    # Feature Importance from Gradient Boosting Classifier
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = X.columns[indices[:24]].tolist()  # Select top 24 features only

    # Set up a dictionary to store the selected features for each 'n'
    selected_features_for_each_n = {}

    # Iterate over the range of 'n' values
    for n in range(6, 25):  # This will iterate from 6 to 24 inclusive
        # SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        shap_feature_importance = np.mean(np.abs(shap_values.values), axis=0)

        # Get indices of features sorted by importance
        shap_indices = np.argsort(shap_feature_importance)[::-1]

        # Select top 'n' features
        shap_features = X.columns[shap_indices[:n]].tolist()

        # Store the selected features in the dictionary
        selected_features_for_each_n[n] = shap_features

    # Return selected features from each method in a dictionary
    selected_features = {
        "RFECV": rfecv_features,
        "GradientBoosting": sorted_features,
        "SHAP": selected_features_for_each_n,
    }
    return selected_features


def fit_and_evaluate(X, y, selected_features, estimator):
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    model = estimator
    for method, features in selected_features.items():
        if method == "SHAP":
            for n, selected_features in features.items():
                scores = []
                for train_index, test_index in tscv.split(X):
                    X_train, X_test = (
                        print("Features used:", features),
                        X.iloc[train_index][selected_features],
                        X.iloc[test_index][selected_features],
                    )
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)

            avg_score = np.mean(scores)
            results[(method, n)] = {
                "features": selected_features,
                "accuracy": avg_score,
            }
    else:  # This will handle "RFECV" and "GradientBoosting"
        scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = (
                print("Features used:", features),
                X.iloc[train_index][features],
                X.iloc[test_index][features],
            )
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)

        avg_score = np.mean(scores)
        results[method] = {
            "features": features,
            "accuracy": avg_score,
        }

    return results


def process_file(i, estimator):
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

    selected_features = apply_feature_selection(X, y, estimator)
    results = fit_and_evaluate(X, y, selected_features, estimator)

    new_results = []
    for key, result in results.items():
        if isinstance(key, tuple):
            # If key is a tuple, unpack it into two variables
            method, num_features = key
        else:
            # If key is not a tuple, it's a string, so use it directly and set num_features to some default
            method = key
            num_features = "N/A"  # Or some other default value

        new_results.append(
            {
                "model": f"Random Forest with {method} Feature Selection ({num_features} features)",
                "duration": i,
                "features": result["features"],
                "accuracy": result["accuracy"],
            }
        )

    return pd.DataFrame(new_results)


if __name__ == "__main__":
    # Instantiate the Gradient Boosting Classifier here
    gb_model = XGBClassifier()

    all_results_list = Parallel(n_jobs=-1)(
        delayed(process_file)(i, gb_model)  # Pass gb_model as an argument
        for i in tqdm(range(60, 61, 3), desc="Processing files")
    )

    # Combine the results from all parallel tasks
    all_results = pd.concat(all_results_list, ignore_index=True)

    # Save the final results DataFrame to a CSV file
    if not all_results.empty:
        all_results.to_csv("gb_minmax.csv", index=False)
        print("All results saved to 'gb_minmax.csv'")
    else:
        print("No results to save.")
