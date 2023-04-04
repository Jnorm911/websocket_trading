models = [
    {"name": "Linear Regression", "model": linear_regression(), "parameters": {...}},
    {"name": "Ridge Regression", "model": ridge_regression(), "parameters": {...}},
    {"name": "Lasso Regression", "model": lasso_regression(), "parameters": {...}},
    {
        "name": "Support Vector Regression",
        "model": support_vector_regression(),
        "parameters": {...},
    },
    {
        "name": "Random Forest Regressor",
        "model": random_forest_regressor(),
        "parameters": {...},
    },
    {
        "name": "Gradient Boosting Regressor",
        "model": gradient_boosting_regressor(),
        "parameters": {...},
    },
    {"name": "XGBoost Regressor", "model": xgboost_regressor(), "parameters": {...}},
    {"name": "LightGBM Regressor", "model": lightgbm_regressor(), "parameters": {...}},
    {"name": "CatBoost Regressor", "model": catboost_regressor(), "parameters": {...}},
    {
        "name": "Multilayer Perceptron",
        "model": multilayer_perceptron(),
        "parameters": {...},
    },
    {"name": "Long Short-Term Memory", "model": lstm(), "parameters": {...}},
    {"name": "Gated Recurrent Unit", "model": gru(), "parameters": {...}},
    {"name": "ARIMA", "model": arima(), "parameters": {...}},
    {"name": "Prophet", "model": prophet(), "parameters": {...}},
    {"name": "Exponential Smoothing", "model": exp_smooth(), "parameters": {...}},
]
