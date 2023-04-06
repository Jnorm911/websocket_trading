import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


## TTV Test, Train, & Validation ##

# Load the preprocessed data for a specific candle length (replace 'i' with the desired value)
i = 59  # or any other value between 1 and 60
preprocessed_data_path = f"data/kc/btc/heiken_ashi/with_trade_indicators/processed/kc_btc_{i}min_ha_ti_pro.csv"
df = pd.read_csv(preprocessed_data_path)


# Predicting the next candle color #
# Define your target variable (column) and remove it from the features list
target_column = "color_change"
X = df.drop(columns=[target_column])
y = df[target_column]


# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create a validation set (further split the training set)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)


print("Training, validation, and test sets created.")

models = [
    # Regression models
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge()),
    ("Lasso Regression", Lasso()),
    ("Support Vector Regression", SVR()),
    ("Random Forest Regressor", RandomForestRegressor(random_state=42)),
    ("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42)),
    ("XGBoost Regressor", XGBRegressor(random_state=42)),
    ("LightGBM Regressor", LGBMRegressor(random_state=42)),
    ("CatBoost Regressor", CatBoostRegressor(random_state=42)),
    ("Multilayer Perceptron", MLPRegressor(random_state=42)),
]

# Train and evaluate the models using cross-validation
model_results = []

for idx, (name, model) in enumerate(models):
    print(f"Training and evaluating model {idx + 1}/{len(models)}: {name}")

    try:
        # Train the model using cross-validation (using 5-fold CV)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_scores_mean = np.mean(cv_scores)

        # Store the results
        model_results.append((name, cv_scores_mean))

        # Print the current model's cross-validation score
        print(f"{name} cross-validation score: {cv_scores_mean}\n")

    except Exception as e:
        print(f"Error occurred while training {name}: {e}\n")
        continue

# Display the results
print("Cross-validation scores for all models:")
for name, score in model_results:
    print(f"{name}: {score}")

# Choose the best model based on the cross-validation scores
best_model_name, best_model_score = max(model_results, key=lambda x: x[1])

print(f"\nBest model: {best_model_name} with score: {best_model_score}")

# Train the best model on the combined training and validation set
best_model = [model for name, model in models if name == best_model_name][0]
best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

# Evaluate the best model on the test set
test_score = best_model.score(X_test, y_test)
print(f"Test accuracy of the best model ({best_model_name}): {test_score}")
