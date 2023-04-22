import concurrent.futures

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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
        scores = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    calculate_score,
                    feature,
                    X_train,
                    y_train,
                    base_estimator,
                    selected_features,
                )
                for feature in remaining_features
            ]

            for future in concurrent.futures.as_completed(futures):
                scores.append(future.result())

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
    model_lr = LogisticRegression(random_state=42, solver="sag", max_iter=10000)
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
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42),
        "KNeighbors": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "GaussianNB": GaussianNB(),
    }

    for method, features in selected_features.items():
        print(f"Features selected by {method}: {features}")  # Debugging print statement
        X_selected = X[features]

        if not X_selected.empty:  # Check if X_selected is not empty
            for model_name, model in models.items():
                cv_scores = cross_val_score(model, X_selected, y, cv=n_splits)
                mean_cv_score = cv_scores.mean()

                results[f"{method}_{model_name}"] = mean_cv_score
        else:
            print(f"No features selected by {method}. Skipping model fitting.")

    return results


# Initialize an empty DataFrame to store the results
all_results = pd.DataFrame(columns=["model", "duration", "best_features"])

# Loop through each file, apply feature selection, fit models, and store the results
for i in range(3, 61, 3):
    print(f"Processing file {i} of 60")
    file_path = rf"data\kc\btc\heiken_ashi\with_trade_indicators\minmax\kline\kc_btc_{i}min_ha_ti_pro.csv"
    data = pd.read_csv(file_path)
    target = "color_change"
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = cross_val_score(
        X, y, test_size=0.2, random_state=42
    )

    print("Applying feature selection...")
    selected_features = apply_feature_selection(X_train, y_train)
    print("Feature selection complete.")

    print("Fitting and evaluating models...")
    results = fit_and_evaluate(X, y, selected_features)
    print("Model fitting and evaluation complete.")

    # Append the results to the all_results DataFrame
    for model, accuracy in results.items():
        new_row = pd.DataFrame(
            {
                "model": [model],
                "duration": [i],
                "best_features": [selected_features[model.split("_")[0]]],
                "accuracy": [accuracy],  # Add the accuracy column here
            }
        )
        all_results = pd.concat([all_results, new_row], ignore_index=True)

    print(f"Results for file {i} appended to the DataFrame")

# Save the final results DataFrame to a CSV file
all_results.to_csv("linear_mm_results.csv", index=False)
print("All results saved to 'all_results.csv'")
