import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNetCV, RidgeCV
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

    # Elastic Net
    print("Applying Elastic Net...")
    elastic_net = ElasticNetCV(
        cv=5,
        random_state=42,
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
        n_alphas=100,
        alphas=None,
    )
    elastic_net.fit(X_train, y_train)
    elastic_net_selector = SelectFromModel(elastic_net)
    elastic_net_selector.fit(X_train, y_train)

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
    ridge = RidgeCV(cv=5, alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100])
    ridge.fit(X_train, y_train)
    ridge_selector = SelectFromModel(
        ridge, prefit=True, max_features=10
    )  # Limit to top 10 features
    ridge_selected_features = X_train.columns[ridge_selector.get_support()].tolist()

    print("Feature selection methods complete.")

    return {
        "RFE": X_train.columns[rfe_selector.support_].tolist(),
        "SelectKBest": X_train.columns[kbest_selector.get_support()].tolist(),
        "ElasticNet": X_train.columns[elastic_net_selector.get_support()].tolist(),
        "LogisticRegression": X_train.columns[lr_selector.get_support()].tolist(),
        "XGBoost": X_train.columns[xgb_selector.get_support()].tolist(),
        "LightGBM": X_train.columns[lgbm_selector.get_support()].tolist(),
        "CatBoost": X_train.columns[cat_selector.get_support()].tolist(),
        "Ridge": ridge_selected_features,
    }


# Function to fit and evaluate models
def fit_and_evaluate(X_train, X_test, y_train, y_test, selected_features):
    results = {}

    for method, features in selected_features.items():
        print(f"Features selected by {method}: {features}")  # Debugging print statement
        X_train_selected = X_train[features]
        X_test_selected = X_test[features]

        if not X_train_selected.empty:  # Check if X_train_selected is not empty
            model_rf = RandomForestClassifier(random_state=42)
            model_rf.fit(X_train_selected, y_train)
            y_pred_rf = model_rf.predict(X_test_selected)
            accuracy_rf = accuracy_score(y_test, y_pred_rf)

            results[f"{method}_RandomForest"] = accuracy_rf
        else:
            print(f"No features selected by {method}. Skipping model fitting.")

    return results


# Initialize an empty DataFrame to store the results
all_results = pd.DataFrame(columns=["model", "duration", "best_features"])

# Loop through each file, apply feature selection, fit models, and store the results
for i in range(3, 61, 3):
    print(f"Processing file {i} of 60")
    file_path = rf"src\models\processed\kc_btc_3min_ha_ti_pro.csv"
    # file_path = f"processed/kc_btc_{i}min_ha_ti_pro.csv"
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
            }
        )
        all_results = pd.concat([all_results, new_row], ignore_index=True)

    print(f"Results for file {i} appended to the DataFrame")

# Save the final results DataFrame to a CSV file
all_results.to_csv("all_results.csv", index=False)
print("All results saved to 'all_results.csv'")
