import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# Function to apply feature selection and return the best features for each method
def apply_feature_selection(X_train, y_train):
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
    model_lr = LogisticRegression(random_state=42)
    model_lr.fit(X_train, y_train)
    lr_selector = SelectFromModel(model_lr)
    lr_selector.fit(X_train, y_train)

    # XGBoost
    print("Applying XGBoost...")
    model_xgb = XGBClassifier(random_state=42)
    model_xgb.fit(X_train, y_train)
    xgb_selector = SelectFromModel(model_xgb)
    xgb_selector.fit(X_train, y_train)

    # LightGBM
    print("Applying LightGBM...")
    model_lgbm = LGBMClassifier(random_state=42)
    model_lgbm.fit(X_train, y_train)
    lgbm_selector = SelectFromModel(model_lgbm)
    lgbm_selector.fit(X_train, y_train)

    # CatBoost
    print("Applying CatBoost...")
    model_cat = CatBoostClassifier(random_state=42, verbose=0)
    model_cat.fit(X_train, y_train)
    cat_selector = SelectFromModel(model_cat)
    cat_selector.fit(X_train, y_train)

    # Ridge
    print("Applying Ridge...")
    # Drop the "time" column
    X_train_no_time = X_train.drop(columns=["time"])
    ridge = RidgeClassifierCV(cv=5, alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100])
    ridge_selector = SelectFromModel(ridge, prefit=True)
    ridge.fit(X_train_no_time, y_train)

    print("Feature selection methods complete.")

    return {
        "RFE": X_train.columns[rfe_selector.support_].tolist(),
        "SelectKBest": X_train.columns[kbest_selector.get_support()].tolist(),
        "LogisticRegression": X_train.columns[lr_selector.get_support()].tolist(),
        "XGBoost": X_train.columns[xgb_selector.get_support()].tolist(),
        "LightGBM": X_train.columns[lgbm_selector.get_support()].tolist(),
        "CatBoost": X_train.columns[cat_selector.get_support()].tolist(),
        "Ridge": X_train_no_time.columns[ridge_selector.get_support()].tolist(),
    }


def fit_and_evaluate(X_train, X_test, y_train, y_test, selected_features):
    results = {}
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
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
all_results = pd.DataFrame(columns=["model", "duration", "best_features"])

# Loop through each file, apply feature selection, fit models, and store the results
for i in range(3, 61, 3):
    print(f"Processing file {i} of 60")
    file_path = rf"data\kc\btc\heiken_ashi\with_trade_indicators\processed\kc_btc_{i}min_ha_ti_pro.csv"
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
                "accuracy": [accuracy],  # Add the accuracy column here
            }
        )
        all_results = pd.concat([all_results, new_row], ignore_index=True)

    print(f"Results for file {i} appended to the DataFrame")

# Save the final results DataFrame to a CSV file
all_results.to_csv("all_results.csv", index=False)
print("All results saved to 'all_results.csv'")
