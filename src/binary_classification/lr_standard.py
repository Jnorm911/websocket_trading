import os

import pandas as pd
from sklearnex import patch_sklearn
from tqdm.auto import tqdm

patch_sklearn()
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection import RFE, RFECV, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def apply_feature_selection(X, y):
    model = LogisticRegression(solver="saga", random_state=42)

    # Add RFE feature selection
    rfe_selector, rfe_features, _ = custom_rfe_feature_selection(X, y, model, 6, 24)

    # Recursive Feature Elimination with Cross-Validation (RFECV)
    rfecv_selector, rfecv_features, _ = custom_rfecv_feature_selection(X, y, model)

    # SelectKBest with mutual information
    kbest_selector, kbest_features = custom_select_kbest_feature_selection(
        X, y, model, k=10
    )

    selected_features = {
        "RFECV": rfecv_features,
        "SelectKBest": kbest_features,
        "RFE": rfe_features,
    }

    return selected_features


def fit_and_evaluate(X, y, selected_features):
    results = {}
    tscv = TimeSeriesSplit(n_splits=5)

    for method in selected_features.keys():
        model = LogisticRegression(
            solver="saga", max_iter=20000, tol=1e-4, random_state=42
        )
        X_selected = X[selected_features[method]]

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        roc_auc_scores = []
        mse_scores = []
        mae_scores = []

        for train_index, test_index in tscv.split(X_selected):
            X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred))
            roc_auc_scores.append(roc_auc_score(y_test, y_pred_proba))
            mse_scores.append(mean_squared_error(y_test, y_pred))
            mae_scores.append(mean_absolute_error(y_test, y_pred))

        results[method] = {
            "best_features": selected_features[method],
            "accuracy": np.mean(accuracy_scores),
            "precision": np.mean(precision_scores),
            "recall": np.mean(recall_scores),
            "roc_auc": np.mean(roc_auc_scores),
            "mse": np.mean(mse_scores),
            "mae": np.mean(mae_scores),
        }

        print(f"{method} feature selection:")
        print(f"Accuracy: {np.mean(accuracy_scores)}")
        print(f"Precision: {np.mean(precision_scores)}")
        print(f"Recall: {np.mean(recall_scores)}")
        print(f"ROC-AUC: {np.mean(roc_auc_scores)}")
        print(f"MSE: {np.mean(mse_scores)}")
        print(f"MAE: {np.mean(mae_scores)}")
        print()

    return results


def custom_rfecv_feature_selection(X_train, y_train, estimator):
    rfecv_selector = RFECV(estimator, step=1, cv=5)
    rfecv_selector.fit(X_train, y_train)
    selected_features = X_train.columns[rfecv_selector.get_support()].tolist()

    return rfecv_selector, selected_features, rfecv_selector.n_features_


def custom_select_kbest_feature_selection(X_train, y_train, estimator, k):
    kbest_selector = SelectKBest(mutual_info_classif, k=k)
    kbest_selector.fit(X_train, y_train)
    selected_features = X_train.columns[kbest_selector.get_support()].tolist()

    return kbest_selector, selected_features


def custom_rfe_feature_selection(
    X_train, y_train, estimator, min_features, max_features
):
    best_score = -1
    best_features = None
    best_n_features = None
    best_selector = None

    for n_features in range(min_features, max_features + 1):
        rfe_selector = RFE(estimator, n_features_to_select=n_features, step=1)
        rfe_selector.fit(X_train, y_train)
        selected_features = X_train.columns[rfe_selector.get_support()].tolist()
        score = cross_val_score(
            estimator, X_train[selected_features], y_train, cv=5
        ).mean()

        if score > best_score:
            best_score = score
            best_features = selected_features
            best_n_features = n_features
            best_selector = rfe_selector

    return best_selector, best_features, best_n_features


def preprocess_data(data):
    features = data.drop(columns=["time", "color_change"])

    scaler = StandardScaler()
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
        "standard",
        "kline",
    )
    file_path = os.path.join(FOLDER_PATH, f"kc_btc_{i}min_ha_ti_pro.csv")
    data = pd.read_csv(file_path)

    # Preprocess the data
    data = preprocess_data(data)

    target = "color_change"
    X = data.drop(columns=[target])
    y = data[target]

    # Apply feature selection
    selected_features = apply_feature_selection(X, y)

    # Fit models and evaluate
    results = fit_and_evaluate(X, y, selected_features)

    # Update the all_results DataFrame
    new_results = []
    for method, result in results.items():
        new_results.append(
            {
                "model": f"Logistic Regression with {method} Feature Selection",
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
        all_results.to_csv("lr_standard.csv", index=False)
        print("All results saved to 'lr_standard.csv'")
    else:
        print("No results to save.")
