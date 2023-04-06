import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from hmmlearn import hmm
from prophet import Prophet
from sklearn.model_selection import cross_val_score, train_test_split
from skorch import NeuralNetClassifier  # You'll need to install 'skorch' package
from sktime.classification.interval_based import TimeSeriesForestClassifier
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import torch
from torch.utils.data import Dataset
from skorch.helper import predefined_split


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(self.y.iloc[idx], dtype=torch.long)
        return x, y


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


train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_test, y_test)


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

# lstm_model definition
lstm_model = NeuralNetClassifier(
    module=LSTMClassifier(input_size, hidden_size, num_layers, num_classes),
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    max_epochs=10,
    batch_size=32,
    train_split=predefined_split(val_dataset),  # Use predefined validation dataset
)

# gru_model definition
gru_model = NeuralNetClassifier(
    module=GRUClassifier(input_size, hidden_size, num_layers, num_classes),
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    max_epochs=10,
    batch_size=32,
    train_split=predefined_split(val_dataset),  # Use predefined validation dataset
)
lstm_model.fit(
    train_dataset, y=None
)  # y=None because the target is included in the dataset
gru_model.fit(
    train_dataset, y=None
)  # y=None because the target is included in the dataset

# Define a list of models
models = [
    ("LSTM Classifier", lstm_model),
    ("GRU Classifier", gru_model),
    ("Hidden Markov Model", hmm.GaussianHMM(n_components=2, covariance_type="diag")),
    ("Time Series Forest Classifier", TimeSeriesForestClassifier()),
    ("DTW K-Nearest Neighbors", KNeighborsTimeSeriesClassifier(metric="dtw")),
    ("ARIMA", ARIMA(order=(p, d, q))),
    ("SARIMA", SARIMAX(order=(p, d, q), seasonal_order=(P, D, Q, s))),
    (
        "ETS",
        ExponentialSmoothing(
            trend=None, damped_trend=None, seasonal=None, seasonal_periods=None
        ),
    ),
    ("Facebook Prophet", Prophet()),
    ("VAR", VAR(endog=y)),
    ("Bayesian Structural Time Series", bsts()),
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
