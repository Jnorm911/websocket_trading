import numpy as np
from sklearn.model_selection import GridSearchCV
import itertools
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

file_path = "C:/Users/jnorm/Projects/websocket_trading/data/kc/btc/heiken_ashi/with_trade_indicators/kc_btc_59min_ha_ti.csv"

df = pd.read_csv(file_path)


# Preprocess the data
df = df.dropna()
df["color"] = df["color"].map({"red": 0, "green": 1})

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Add interaction terms and transformations to the dataset
train_df["BBU_BBP"] = train_df["BBU_5_2.0"] * train_df["BBP_5_2.0"]
test_df["BBU_BBP"] = test_df["BBU_5_2.0"] * test_df["BBP_5_2.0"]

train_df["RSI_log"] = np.log(train_df["RSI"])
test_df["RSI_log"] = np.log(test_df["RSI"])

# Update the list of tech_indicators with new features
tech_indicators = [
    "PP",
    "R1",
    "S1",
    "R2",
    "S2",
    "BBL_5_2.0",
    "BBM_5_2.0",
    "BBU_5_2.0",
    "BBB_5_2.0",
    "BBP_5_2.0",
    "RSI",
    "BBU_BBP",
    "RSI_log",
]

# Choose a combination based on your previous best results
combo = [
    "PP",
    "R1",
    "S1",
    "BBL_5_2.0",
    "BBM_5_2.0",
    "BBU_5_2.0",
    "BBB_5_2.0",
    "BBP_5_2.0",
    "RSI",
    "BBU_BBP",
    "RSI_log",
]

X_train = train_df[combo]
y_train = train_df["color"]
X_test = test_df[combo]
y_test = test_df["color"]

# Set up the parameter grid for GridSearchCV
param_grid = {
    "C": np.logspace(-4, 4, 20),
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
}

# Create a GridSearchCV object
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000), param_grid, scoring="accuracy", cv=5
)

# Fit the GridSearchCV object
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train a Logistic Regression model with the best hyperparameters
model = LogisticRegression(
    max_iter=1000, C=best_params["C"], solver=best_params["solver"]
)
model.fit(X_train, y_train)

# Make predictions with the optimized model
y_pred = model.predict(X_test)

# Calculate the improved accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
