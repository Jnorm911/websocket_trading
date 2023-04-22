import os
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()

import ray
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm
from xgboost import XGBClassifier

ray.init()


# Function to apply feature selection and return the best features for each method
def apply_feature_selection(X_train, y_train):
    # Nystroem transformer with LinearSVC
    # Apply SelectKBest before Nystroem transformation
    print("Apply SelectKBest...")
    selector = SelectKBest(f_classif, k=1)  # or any other value for k
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Nystroem transformer with LinearSVC
    print("Apply Nystroem transformer with LinearSVC...")
    nystroem = Nystroem(random_state=42)
    linear_svc = LinearSVC(random_state=42)
    pipeline = make_pipeline(nystroem, linear_svc)
    pipeline.fit(X_train_selected, y_train)

    selected_features_mask = selector.get_support()

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

    # Random Forests
    print("Applying Random Forests...")
    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train, y_train)
    rf_selector = SelectFromModel(model_rf)
    rf_selector.fit(X_train, y_train)

    # Gradient Boosting Machines (GBM)
    print("Applying Gradient Boosting Machines...")
    model_gbm = GradientBoostingClassifier(random_state=42)
    model_gbm.fit(X_train, y_train)
    gbm_selector = SelectFromModel(model_gbm)
    gbm_selector.fit(X_train, y_train)

    # Extra Trees
    print("Applying ExtraTreesClassifier...")
    model_et = ExtraTreesClassifier(random_state=42)
    model_et.fit(X_train, y_train)
    et_selector = SelectFromModel(model_et)
    et_selector.fit(X_train, y_train)

    # KNN
    print("Applying KNN...")
    knn_selector = SelectKBest(f_classif, k=4)
    knn_selector.fit(X_train, y_train)

    print("Feature selection methods complete.")

    return {
        "SVM": X_train.columns[selected_features_mask].tolist(),
        "XGBoost": X_train.columns[xgb_selector.get_support()].tolist(),
        "LightGBM": X_train.columns[lgbm_selector.get_support()].tolist(),
        "CatBoost": X_train.columns[cat_selector.get_support()].tolist(),
        "Random Forests": X_train.columns[rf_selector.get_support()].tolist(),
        "Gradient Boosting Machines": X_train.columns[
            gbm_selector.get_support()
        ].tolist(),
        "ExtraTrees": X_train.columns[et_selector.get_support()].tolist(),
        "KNN": X_train.columns[knn_selector.get_support()],
    }


def fit_and_evaluate(X, y, selected_features, n_splits=5):
    results = {}
    models = {
        "SVM": make_pipeline(Nystroem(random_state=42), LinearSVC(random_state=42)),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
        "LGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "ExtraTrees": ExtraTreesClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
    }

    for method, features in selected_features.items():
        print(f"Features selected by {method}: {features}")
        X_selected = X[features]

        if not X_selected.empty:
            for model_name, model in models.items():
                X_train, X_test, y_train, y_test = X_selected, X_selected, y, y
                X_train_transformed, X_test_transformed = X_train, X_test

                print(f"Fitting {model_name} with features selected by {method}...")
                model.fit(X_train_transformed, y_train)
                print(
                    f"Calculating cross_val_score for {model_name} with features selected by {method}..."
                )
                scores = cross_val_score(
                    model, X_test_transformed, y_test, cv=n_splits, n_jobs=-1
                )
                mean_score = np.mean(scores)
                results[f"{method}_{model_name}"] = mean_score
                print(
                    f"Mean cross_val_score for {model_name} with features selected by {method}: {mean_score}"
                )
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
    for model_name, accuracy in results.items():
        new_results.append(
            {
                "model": model_name,
                "duration": i,
                "best_features": selected_features,
                "accuracy": accuracy,
            }
        )

    return pd.DataFrame(new_results)


if __name__ == "__main__":
    # Loop through each file, apply feature selection, fit models, and store the results
    all_results_list = Parallel(n_jobs=-1)(
        delayed(process_file)(i) for i in tqdm(range(3, 61, 3), desc="Processing files")
    )

    # Filter out None values from the results list
    all_results_list = [result for result in all_results_list if result is not None]

    # Combine the results from all parallel tasks
    all_results = pd.concat(all_results_list, ignore_index=True)

    # Save the final results DataFrame to a CSV file
    all_results.to_csv("tree_robust_results.csv", index=False)
    print("All results saved to 'tree_robust_results.csv'")
