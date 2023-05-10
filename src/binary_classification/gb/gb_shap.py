import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
from joblib import Parallel, delayed
from tqdm import tqdm


def get_top_shap_features(X, model, top_n=10):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_sum = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ["column_name", "shap_importance"]
    importance_df = importance_df.sort_values("shap_importance", ascending=False)
    top_features = importance_df.iloc[:top_n]["column_name"].tolist()
    return top_features


def check_data_balance(y):
    counter = Counter(y)
    for label, count in counter.items():
        percentage = (count / len(y)) * 100
        print(f"Class {label}: {count} samples ({percentage:.2f}%)")


def apply_feature_selection(X, y, model_name, top_n_features=10):
    if model_name == "XGBoost":
        estimator = XGBClassifier(eval_metric="logloss")
    elif model_name == "LightGBM":
        estimator = LGBMClassifier()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Train the estimator
    estimator.fit(X, y)

    # Get top features based on SHAP values
    top_features = get_top_shap_features(X, estimator, top_n=top_n_features)

    # Create a boolean mask for the selected features
    selected_features_mask = X.columns.isin(top_features)

    return selected_features_mask


def fit_and_evaluate(X, y, i, model_name, top_n_features=10):
    # Split the data into train and test sets using TimeSeriesSplit
    cv = TimeSeriesSplit(n_splits=5)

    results = []

    for fold_number, (train_index, test_index) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the selected model
        if model_name == "XGBoost":
            model = XGBClassifier(eval_metric="logloss")
        elif model_name == "LightGBM":
            model = LGBMClassifier()
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        model.fit(X_train, y_train)

        # Get top features based on SHAP values
        top_features = get_top_shap_features(X_train, model, top_n=top_n_features)

        # Select the top features
        X_train_selected = X_train.loc[:, top_features]
        X_test_selected = X_test.loc[:, top_features]

        # Train the model again on the selected features
        model.fit(X_train_selected, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_selected)
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        results.append(
            {
                "model": f"{model_name} with SHAP Feature Selection",
                "duration": i,
                "best_features": ", ".join(top_features),
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
    xgb_results_df = fit_and_evaluate(X, y, i, model_name="XGBoost", top_n_features=10)
    lgbm_results_df = fit_and_evaluate(
        X, y, i, model_name="LightGBM", top_n_features=10
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
        all_results.to_csv("gb_shap.csv", index=False)
        print("All results saved to 'gb_shap.csv'")
    else:
        print("No results to save.")
