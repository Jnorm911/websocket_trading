import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# Function to apply feature selection and return the best features for each method
def apply_feature_selection(X_train, y_train):
    # RFE
    estimator = RandomForestClassifier(random_state=42)
    rfe_selector = RFE(estimator, n_features_to_select=10, step=1)
    rfe_selector.fit(X_train, y_train)

    # SelectKBest
    kbest_selector = SelectKBest(f_classif, k=10)
    kbest_selector.fit(X_train, y_train)

    # LASSO
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_train, y_train)
    lasso_selector = SelectFromModel(lasso)
    lasso_selector.fit(X_train, y_train)

    # Logistic Regression
    model_lr = LogisticRegression(random_state=42)
    model_lr.fit(X_train, y_train)
    lr_selector = SelectFromModel(model_lr)
    lr_selector.fit(X_train, y_train)

    # XGBoost
    model_xgb = XGBClassifier(random_state=42)
    model_xgb.fit(X_train, y_train)
    xgb_selector = SelectFromModel(model_xgb)
    xgb_selector.fit(X_train, y_train)

    # SVC with linear kernel
    model_svc = SVC(kernel="linear", random_state=42)
    model_svc.fit(X_train, y_train)
    svc_selector = SelectFromModel(model_svc)
    svc_selector.fit(X_train, y_train)

    return {
        "RFE": X_train.columns[rfe_selector.support_].tolist(),
        "SelectKBest": X_train.columns[kbest_selector.get_support()].tolist(),
        "LASSO": X_train.columns[lasso_selector.get_support()].tolist(),
        "LogisticRegression": X_train.columns[lr_selector.get_support()].tolist(),
        "XGBoost": X_train.columns[xgb_selector.get_support()].tolist(),
        "SVC": X_train.columns[svc_selector.get_support()].tolist(),
    }


# Function to fit and evaluate models
def fit_and_evaluate(X_train, X_test, y_train, y_test, selected_features):
    results = {}

    for method, features in selected_features.items():
        X_train_selected = X_train[features]
        X_test_selected = X_test[features]

        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(X_train_selected, y_train)
        y_pred_rf = model_rf.predict(X_test_selected)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)

        results[f"{method}_RandomForest"] = accuracy_rf

    return results


# Loop through each file, apply feature selection, fit models, and display the results
for i in range(1, 61):
    file_path = f"data/kc/btc/heiken_ashi/with_trade_indicators/processed/kc_btc_{i}min_ha_ti_pro.csv"
    data = pd.read_csv(file_path)
    target = "color_change"  # Using 'color_change' as the target column
    X = data.drop(columns=[target])  # Drop the target column from the features
    y = data[target]  # Use the target column as the target variable
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    selected_features = apply_feature_selection(X_train, y_train)
    results = fit_and_evaluate(X_train, X_test, y_train, y_test, selected_features)

    display(pd.DataFrame(results, index=[f"File {i}"]).T)
