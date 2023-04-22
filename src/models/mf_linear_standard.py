import concurrent.futures
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LogisticRegression,
    RidgeClassifierCV,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

## USE STANDARD KLINE


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


# Function to apply feature selection and return the best features for each method
def apply_feature_selection(X_train, y_train):
    # RFA
    print("Applying RFA...")
    estimator = RandomForestClassifier(random_state=42)
    rfa_features = rfa(X_train, y_train, estimator)

    # RFE
    print("Applying RFE...")
    estimator = RandomForestClassifier(random_state=42)
    rfe_selector = RFE(estimator, n_features_to_select=10, step=1)
    rfe_selector.fit(X_train, y_train)

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

    # Ridge
    print("Applying Ridge...")
    # Drop the "time" column
    X_train_no_time = X_train.drop(columns=["time"])
    ridge = RidgeClassifierCV(cv=5, alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100])
    ridge_selector = SelectFromModel(ridge, prefit=True)
    ridge.fit(X_train_no_time, y_train)

    # LassoCV
    print("Applying Lasso...")
    lasso_cv = LassoCV(cv=5).fit(X_train, y_train)
    lasso_cv_selector = lasso_cv.coef_ != 0
    lasso_cv_features = X_train.columns[lasso_cv_selector].tolist()

    # ElasticNetCV
    print("Applying ElasticNetCV...")
    elastic_net_cv = ElasticNetCV(cv=5).fit(X_train, y_train)
    elastic_net_cv_selector = elastic_net_cv.coef_ != 0
    elastic_net_cv_features = X_train.columns[elastic_net_cv_selector].tolist()

    print("Feature selection methods complete.")

    return {
        "RFA": rfa_features,
        "RFE": X_train.columns[rfe_selector.support_].tolist(),
        "SelectKBest": X_train.columns[kbest_selector.get_support()].tolist(),
        "LogisticRegression": X_train.columns[lr_selector.get_support()].tolist(),
        "Ridge": X_train_no_time.columns[ridge_selector.get_support()].tolist(),
        "LassoCV": lasso_cv_features,
        "ElasticNetCV": elastic_net_cv_features,
    }


def fit_and_evaluate(X_train, X_test, y_train, y_test, selected_features):
    results = {}
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42),
        "Ridge": RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100]),
    }

    for method, features in selected_features.items():
        print(f"Features selected by {method}: {features}")  # Debugging print statement
        X_train_selected = X_train[features]
        X_test_selected = X_test[features]

        if not X_train_selected.empty:  # Check if X_train_selected is not empty
            for model_name, model in models.items():
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
                accuracy = accuracy_score(y_test, y_pred)

                results[f"{method}_{model_name}"] = accuracy
        else:
            print(f"No features selected by {method}. Skipping model fitting.")

    return results


# Initialize an empty DataFrame to store the results
all_results = pd.DataFrame(columns=["model", "duration", "best_features", "accuracy"])

# Loop through each file, apply feature selection, fit models, and store the results
for i in range(3, 61, 3):
    print(f"Processing file {i} of 60")
    file_path = rf"data\kc\btc\heiken_ashi\with_trade_indicators\standard\kline\kc_btc_{i}min_ha_ti_pro.csv"
    data = pd.read_csv(file_path)
    target = "color_change"
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Applying feature selection...")
    selected_features = apply_feature_selection(X_train, y_train)
    print("Feature selection complete.")

    print("Fitting and evaluating models...")
    results = fit_and_evaluate(X_train, X_test, y_train, y_test, selected_features)
    print("Model fitting and evaluation complete.")

    # Append the results to the all_results DataFrame
    for model, accuracy in results.items():
        new_row = pd.DataFrame(
            {
                "model": [model],
                "duration": [i],
                "best_features": [selected_features[model.split("_")[0]]],
                "accuracy": [accuracy],
            }
        )
        all_results = pd.concat([all_results, new_row], ignore_index=True)

    print(f"Results for file {i} appended to the DataFrame")

# Save the final results DataFrame to a CSV file
all_results.to_csv("linear_standard_results.csv", index=False)
print("All results saved to 'linear_standard_results.csv'")
