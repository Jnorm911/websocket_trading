import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from catboost import CatBoostRegressor
from hmmlearn import hmm
from lightgbm import LGBMRegressor
from prophet import Prophet
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from skorch import NeuralNetClassifier  # You'll need to install 'skorch' package
from sktime.classification.interval_based import TimeSeriesForestClassifier
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from torch.utils.data import DataLoader, TensorDataset
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from xgboost import XGBRegressor


# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Define the GRU model
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


## TTV Test, Train, & Validation ##

# Load the preprocessed data for a specific candle length (replace 'i' with the desired value)
i = 60  # or any other value between 1 and 60
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

# Replace these values with the appropriate ones for your dataset
input_size = X_train.shape[1]  # Calculate the number of features from X_train
hidden_size = 50
num_layers = 1
num_classes = 2

lstm_model = NeuralNetClassifier(
    module=LSTMClassifier(input_size, hidden_size, num_layers, num_classes),
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    max_epochs=10,
    batch_size=32,
    train_split=None,  # Disable internal train/validation split
)

gru_model = NeuralNetClassifier(
    module=GRUClassifier(input_size, hidden_size, num_layers, num_classes),
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    max_epochs=10,
    batch_size=32,
    train_split=None,  # Disable internal train/validation split
)


# Define a list of models
models = [
    # Classification models
    # ("Logistic Regression", LogisticRegression(random_state=42)),
    ("Random Forest Classifier", RandomForestClassifier(random_state=42)),
    ("Gradient Boosting Classifier", GradientBoostingClassifier(random_state=42)),
    # ("Support Vector Classification", SVC(kernel="rbf", random_state=42)),
    # ("k-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
    ("Decision Tree Classifier", DecisionTreeClassifier(random_state=42)),
    # ("Gaussian Naive Bayes", GaussianNB()),
    # ("Multinomial Naive Bayes", MultinomialNB()),
    # ("Bernoulli Naive Bayes", BernoulliNB()),
    # ("Multilayer Perceptron Classifier", MLPClassifier(random_state=42)),
    # ("Stochastic Gradient Descent Classifier", SGDClassifier(random_state=42)),
    ("Extra Trees Classifier", ExtraTreesClassifier(random_state=42)),
    # Time Series
    ("Hidden Markov Model", hmm.GaussianHMM(n_components=2, covariance_type="diag")),
    ("LSTM Classifier", lstm_model),
    ("GRU Classifier", gru_model),
    ("Time Series Forest Classifier", TimeSeriesForestClassifier()),
    # ("DTW K-Nearest Neighbors", KNeighborsTimeSeriesClassifier(metric="dtw")),
    # Regression models
    # ("Linear Regression", LinearRegression()),
    # ("Ridge Regression", Ridge()),
    # ("Lasso Regression", Lasso()),
    # ("Support Vector Regression", SVR()),
    # ("Random Forest Regressor", RandomForestRegressor(random_state=42)),
    # ("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42)),
    # ("XGBoost Regressor", XGBRegressor(random_state=42)),
    # ("LightGBM Regressor", LGBMRegressor(random_state=42)),
    # ("CatBoost Regressor", CatBoostRegressor(random_state=42)),
    # ("Multilayer Perceptron", MLPRegressor(random_state=42)),
    # ("ARIMA", ARIMA(endog=y_train, order=(1, 0, 0))),
    # ("Exponential Smoothing", ExponentialSmoothing(y_train)),
    # Custom implementation models
    # ("Prophet", prophet()),
    # ("Grid Search CV", grid_search_cv()),
    # ("Ensemble", ensemble()),
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
