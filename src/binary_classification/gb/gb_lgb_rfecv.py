import os

import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()
from collections import Counter

from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from xgboost import XGBClassifier


def check_data_balance(y):
    counter = Counter(y)
    for label, count in counter.items():
        percentage = (count / len(y)) * 100
        print(f"Class {label}: {count} samples ({percentage:.2f}%)")


def apply_feature_selection(X, y, model_name):
    if model_name == "XGBoost":
        estimator = XGBClassifier(eval_metric="logloss")
    elif model_name == "LightGBM":
        estimator = LGBMClassifier()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    cv = TimeSeriesSplit(n_splits=5)
    selector = RFECV(estimator, step=1, cv=cv, scoring="f1")
    selector = selector.fit(X, y)
    return selector.support_


def fit_and_evaluate(X, y, selected_features, i, model_name):
    X_selected = X.loc[:, selected_features]

    # Split the data into train and test sets using TimeSeriesSplit
    cv = TimeSeriesSplit(n_splits=5)

    results = []

    for fold_number, (train_index, test_index) in enumerate(cv.split(X_selected)):
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the selected model
        if model_name == "XGBoost":
            model = XGBClassifier(eval_metric="logloss")
        elif model_name == "LightGBM":
            model = LGBMClassifier()
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        results.append(
            {
                "model": f"{model_name} with RFECV Feature Selection",
                "duration": i,
                "best_features": ", ".join(X.columns[selected_features]),
                "fold": fold_number + 1,  # Adding fold column, starting from 1
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "auc_roc": auc_roc,
            }
        )

    return pd.DataFrame(results)


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

    # Check if data is balanced or unbalanced
    print(f"Data balance for duration {i} minutes:")
    check_data_balance(y)

    # Apply feature selection for XGBoost
    xgb_selected_features = apply_feature_selection(X, y, model_name="XGBoost")

    # Apply feature selection for LightGBM
    lgbm_selected_features = apply_feature_selection(X, y, model_name="LightGBM")

    # Fit models and evaluate
    xgb_results_df = fit_and_evaluate(
        X, y, xgb_selected_features, i, model_name="XGBoost"
    )
    lgbm_results_df = fit_and_evaluate(
        X, y, lgbm_selected_features, i, model_name="LightGBM"
    )

    return pd.concat([xgb_results_df, lgbm_results_df], ignore_index=True)


if __name__ == "__main__":
    # Loop through each file, apply feature selection, fit models, and store the results
    all_results_list = Parallel(n_jobs=-1)(
        delayed(process_file)(i) for i in tqdm(range(3, 61, 3), desc="Processing files")
    )

    # Combine the results from all parallel tasks
    all_results = pd.concat(all_results_list, ignore_index=True)

    # Save the final results DataFrame to a CSV file
    if not all_results.empty:
        all_results.to_csv("gb_rfecv.csv", index=False)
        print("All results saved to 'gb_rfecv.csv'")
    else:
        print("No results to save.")
