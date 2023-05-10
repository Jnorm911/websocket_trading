import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class KLineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BinaryClassificationNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryClassificationNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x.squeeze()


def apply_feature_selection(X, y):
    # Define the models for feature selection
    # Define the models for feature selection
    lasso_model = Lasso(alpha=0.001, max_iter=10000, tol=0.01, random_state=42)
    elastic_net_model = ElasticNet(
        l1_ratio=0.5, alpha=0.001, max_iter=10000, tol=0.01, random_state=42
    )
    # Increase the alpha parameter value
    ridge_model = Ridge(alpha=5, random_state=42)

    # Lasso Regularization (L1)
    lasso_selector = SelectFromModel(lasso_model)
    lasso_selector.fit(X, y)
    lasso_features = X.columns[lasso_selector.get_support()]

    # Ridge Regularization (L2)
    ridge_selector = SelectFromModel(ridge_model)
    ridge_selector.fit(X, y)
    ridge_features = X.columns[ridge_selector.get_support()]

    # Elastic Net Regularization (L1+L2)
    elastic_net_selector = SelectFromModel(elastic_net_model)
    elastic_net_selector.fit(X, y)
    elastic_net_features = X.columns[elastic_net_selector.get_support()]

    selected_features = {
        "Lasso": lasso_features,
        "Ridge": ridge_features,
        "ElasticNet": elastic_net_features,
    }

    return selected_features


def train_and_evaluate(
    X,
    y,
    selected_features,
    regularization=None,
    l1_lambda=0.001,
    l2_lambda=0.001,
    epochs=50,
    batch_size=64,
):
    def l1_penalty(parameters):
        return sum(torch.sum(torch.abs(param.view(-1))) for param in parameters)

    def l2_penalty(parameters):
        return sum(torch.sum(param.view(-1) ** 2) for param in parameters)

    X_selected = X[selected_features]

    # Split the dataset
    tscv = TimeSeriesSplit(n_splits=5)
    train_index, test_index = list(tscv.split(X_selected))[-1]
    X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Create datasets and data loaders
    train_dataset = KLineDataset(X_train, y_train)
    test_dataset = KLineDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_size = X_selected.shape[1]
    hidden_size = 32
    output_size = 1
    model = BinaryClassificationNN(input_size, hidden_size, output_size)
    l1_lambda = 0.001
    l2_lambda = 0.001
    l1_loss = l1_penalty(model.parameters()) * l1_lambda
    l2_loss = l2_penalty(model.parameters()) * l2_lambda

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            if regularization == "l1":
                l1_loss = l1_penalty(model.parameters()) * l1_lambda
                loss += l1_loss
            elif regularization == "l2":
                l2_loss = l2_penalty(model.parameters()) * l2_lambda
                loss += l2_loss
            elif regularization == "elastic_net":
                l1_loss = l1_penalty(model.parameters()) * l1_lambda
                l2_loss = l2_penalty(model.parameters()) * l2_lambda
                loss += l1_loss + l2_loss

            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    y_pred = []
    y_pred_proba = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            y_pred_proba.extend(outputs.detach().numpy().tolist())
            y_pred.extend(np.round(outputs.detach().numpy().tolist()))

    # Add selected features to the output
    return {
        "selected_features": ", ".join(selected_features.tolist()),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred),
        "recall_score": recall_score(y_test, y_pred),
        "roc_auc_score": roc_auc_score(y_test, y_pred_proba),
    }


def process_file(i):
    FOLDER_PATH = os.path.join(
        "data",
        "kc",
        "btc",
        "heiken_ashi",
        "with_trade_indicators",
        "standard",
        "kline",
    )
    file_path = os.path.join(FOLDER_PATH, f"kc_btc_{i}min_ha_ti_pro.csv")
    data = pd.read_csv(file_path)

    # Preprocess the data
    data = preprocess_data(data)

    target = "color_change"
    X = data.drop(columns=[target])
    y = data[target]

    # Check if data is balanced or unbalanced
    print(f"Data balance for duration {i} minutes:")
    check_data_balance(y)

    # Apply feature selections
    selected_features = apply_feature_selection(X, y)

    # Store the selected features in a DataFrame or a CSV file
    df_selected_features = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in selected_features.items()])
    )
    df_selected_features.to_csv("selected_features.csv")

    # Initialize the results list
    all_results = []

    # Fit the MLP model with L1 regularization
    l1_results = train_and_evaluate(
        X[selected_features["Lasso"]],
        y,
        selected_features=selected_features["Lasso"],
        regularization="l1",
        l1_lambda=0.001,
    )
    l1_results["model"] = "MLP with L1 Regularization"
    l1_results["duration"] = i
    all_results.append(l1_results)

    # Fit the MLP model with L2 regularization
    l2_results = train_and_evaluate(
        X[selected_features["Ridge"]],
        y,
        selected_features=selected_features["Ridge"],
        regularization="l2",
        l2_lambda=0.001,
    )

    l2_results["model"] = "MLP with L2 Regularization"
    l2_results["duration"] = i
    all_results.append(l2_results)

    # Fit the MLP model with Elastic Net regularization
    elastic_net_results = train_and_evaluate(
        X[selected_features["ElasticNet"]],
        y,
        selected_features=selected_features["ElasticNet"],
        regularization="elastic_net",
        l1_lambda=0.001,
        l2_lambda=0.001,
    )

    elastic_net_results["model"] = "MLP with Elastic Net Regularization"
    elastic_net_results["duration"] = i
    all_results.append(elastic_net_results)

    # Create the DataFrame and specify the order of the columns
    column_order = [
        "model",
        "duration",
        "selected_features",
        "accuracy",
        "f1_score",
        "precision_score",
        "recall_score",
        "roc_auc_score",
    ]
    results_df = pd.DataFrame(all_results)[column_order]

    return results_df


def check_data_balance(y):
    counter = Counter(y)
    for label, count in counter.items():
        percentage = (count / len(y)) * 100
        print(f"Class {label}: {count} samples ({percentage:.2f}%)")


def preprocess_data(data):
    features = data.drop(columns=["time", "color_change"])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    data[features.columns] = scaled_features

    return data


if __name__ == "__main__":
    # Loop through each file, apply feature selection, fit models, and store the results
    all_results_list = Parallel(n_jobs=-1)(
        delayed(process_file)(i) for i in tqdm(range(3, 61, 3), desc="Processing files")
    )

    # Combine the results from all parallel tasks
    all_results = pd.concat(all_results_list, ignore_index=True)

    # Save the final results DataFrame to a CSV file
    if not all_results.empty:
        all_results.to_csv("nn_l1.csv", index=False)
        print("All results saved to 'nn.csv'")
    else:
        print("No results to save.")
