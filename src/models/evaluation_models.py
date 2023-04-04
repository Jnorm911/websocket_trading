import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
from quick_call_ttv import load_preprocessed_data
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor

# LSTM and GRU are not directly importable from a library
# You'll need to create custom classes using TensorFlow or PyTorch for these models

models = [
    {"name": "Linear Regression", "model": LinearRegression(), "parameters": {...}},
    {"name": "Ridge Regression", "model": Ridge(), "parameters": {...}},
    {"name": "Lasso Regression", "model": Lasso(), "parameters": {...}},
    {"name": "Support Vector Regression", "model": SVR(), "parameters": {...}},
    {
        "name": "Random Forest Regressor",
        "model": RandomForestRegressor(),
        "parameters": {...},
    },
    {
        "name": "Gradient Boosting Regressor",
        "model": GradientBoostingRegressor(),
        "parameters": {...},
    },
    {"name": "XGBoost Regressor", "model": XGBRegressor(), "parameters": {...}},
    {"name": "LightGBM Regressor", "model": LGBMRegressor(), "parameters": {...}},
    {"name": "CatBoost Regressor", "model": CatBoostRegressor(), "parameters": {...}},
    {"name": "Multilayer Perceptron", "model": MLPRegressor(), "parameters": {...}},
    # {"name": "Long Short-Term Memory", "model": lstm(), "parameters": {...}},
    # {"name": "Gated Recurrent Unit", "model": gru(), "parameters": {...}},
    {
        "name": "ARIMA",
        "model": ARIMA(endog=y_train, order=(1, 0, 0)),
        "parameters": {...},
    },
    # {"name": "Prophet", "model": prophet(), "parameters": {...}}, # Requires additional preprocessing steps
    {
        "name": "Exponential Smoothing",
        "model": ExponentialSmoothing(y_train),
        "parameters": {...},
    },
    # {"name": "Grid Search CV", "model": grid_search_cv(), "parameters": {...}},
    # {"name": "Ensemble", "model": ensemble(), "parameters": {...}},
]

# quick_call_ttv.py


def load_preprocessed_data(candle_length):
    preprocessed_data_path = f"data/kc/btc/heiken_ashi/with_trade_indicators/processed/kc_btc_{candle_length}min_ha_ti_pro.csv"
    df = pd.read_csv(preprocessed_data_path)

    target_column = "color_green"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )

    return X_train, X_val, y_train, y_val, X_test, y_test


# Function to evaluate models
def evaluate_models(models, candle_length):
    X_train, X_val, y_train, y_val, X_test, y_test = load_preprocessed_data(
        candle_length
    )
    model_results = []

    for model_info in models:
        model = model_info["model"]
        model_name = model_info["name"]
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        # You may need to round predictions to 0 or 1 for classification models
        train_accuracy = accuracy_score(y_train, (train_predictions > 0.5).astype(int))
        val_accuracy = accuracy_score(y_val, (val_predictions > 0.5).astype(int))

        model_results.append(
            {
                "model": model_name,
                "train_accuracy": train_accuracy,
                "validation_accuracy": val_accuracy,
            }
        )
        print(
            f"{model_name} - Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

    return model_results


# Evaluate models for a specific candle length (e.g., 59 minutes)
candle_length = 59
results = evaluate_models(models, candle_length)
