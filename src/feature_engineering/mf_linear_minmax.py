import os

import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

## USE MINMAX KLINE


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


def calculate_score(feature, X_train, y_train, base_estimator, selected_features):
    estimator_copy = clone(base_estimator)
    test_features = selected_features + [feature]
    estimator_copy.fit(X_train[test_features], y_train)
    score = estimator_copy.score(X_train[test_features], y_train)
    return (feature, score)


def rfa(X_train, y_train, base_estimator, step=1):
    remaining_features = X_train.columns.tolist()
    selected_features = []

    while remaining_features:
        scores = Parallel(n_jobs=-1)(
            delayed(calculate_score)(
                feature, X_train, y_train, base_estimator, selected_features
            )
            for feature in remaining_features
        )

        best_features = sorted(scores, key=lambda x: x[1], reverse=True)[:step]
        selected_features.extend([x[0] for x in best_features])

        for feature, _ in best_features:
            remaining_features.remove(feature)

    return selected_features


def apply_feature_selection(X_train, y_train):
    # RFA
    print("Applying RFA...")
    estimator = RandomForestClassifier(random_state=42)
    rfa_features = rfa(X_train, y_train, estimator)

    # # RFE
    print("Applying RFE...")
    estimator = RandomForestClassifier(random_state=42)
    min_features = 5
    max_features = 20
    best_selector, best_features, best_n_features = custom_rfe_feature_selection(
        X_train, y_train, estimator, min_features, max_features
    )

    # SelectKBest
    print("Applying SelectKBest...")
    kbest_selector = SelectKBest(f_classif, k=10)
    kbest_selector.fit(X_train, y_train)

    # Logistic Regression
    print("Applying Logistic Regression...")
    model_lr = LogisticRegression(random_state=42, solver="liblinear", max_iter=10000)
    model_lr.fit(X_train, y_train)
    lr_selector = SelectFromModel(model_lr)
    lr_selector.fit(X_train, y_train)

    print("Feature selection methods complete.")

    return {
        "RFA": rfa_features,
        "RFE": best_features,
        "SelectKBest": X_train.columns[kbest_selector.get_support()].tolist(),
        "LogisticRegression": X_train.columns[lr_selector.get_support()].tolist(),
    }


def fit_and_evaluate(X, y, selected_features, n_splits=5):
    results = {}
    models = {
        # "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(
            random_state=42, solver="liblinear", max_iter=10000
        ),
        # "KNeighbors": KNeighborsClassifier(),
        # "DecisionTree": DecisionTreeClassifier(random_state=42),
        # "GaussianNB": GaussianNB(),
    }

    for method, features in selected_features.items():
        print(f"Features selected by {method}: {features}")  # Debugging print statement
        X_selected = X[features]

        if not X_selected.empty:  # Check if X_selected is not empty
            for model_name, model in models.items():
                cv_scores = cross_val_score(model, X_selected, y, cv=n_splits)
                mean_cv_score = cv_scores.mean()

                results[f"{method}_{model_name}"] = {
                    "best_features": features,
                    "mean_score": mean_cv_score,
                }
        else:
            print(f"No features selected by {method}. Skipping model fitting.")

    return results


def process_file(i):
    FOLDER_PATH = os.path.join(
        "data",
        "kc",
        "btc",
        "heiken_ashi",
        "with_trade_indicators",
        "robust",
        "kline",
    )
    file_path = os.path.join(FOLDER_PATH, f"kc_btc_{i}min_ha_ti_pro.csv")
    data = pd.read_csv(file_path)
    target = "color_change"
    X = data.drop(columns=[target])
    y = data[target]

    # Apply feature selection
    selected_features = apply_feature_selection(X, y)

    # Fit models and evaluate
    results = fit_and_evaluate(X, y, selected_features)

    # Update the all_results DataFrame
    new_results = []
    for model_name, result in results.items():
        new_results.append(
            {
                "model": model_name,
                "duration": i,
                "best_features": result["best_features"],
                "accuracy": result["mean_score"],
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
        all_results.to_csv("linear_minmax_results3.csv", index=False)
        print("All results saved to 'linear_minmax_results3.csv'")
    else:
        print("No results to save.")
