import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ThreadPoolExecutor


# Define the function to evaluate a single model
def evaluate_model(name, model):
    best_score = 0
    best_features = []

    # Loop through all possible feature combinations
    for n in range(1, len(features) + 1):
        for subset in itertools.combinations(features, n):
            # Prepare the data for the current feature subset
            X_subset = X[list(subset)]

            # Train the model and calculate the cross-validation score
            cv_scores = cross_val_score(model, X_subset, y, cv=5, scoring="accuracy")
            cv_scores_mean = np.mean(cv_scores)

            # Print the current model, subset, and cross-validation score
            print(f"Model: {name}" f" Subset: {subset}", f" Score: {cv_scores_mean}")

            # Update the best score and features if the current score is better
            if cv_scores_mean > best_score:
                best_score = cv_scores_mean
                best_features = subset

    return {
        "Model": name,
        "Best Score": best_score,
        "Best Features": best_features,
    }


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

# Define columns to exclude
exclude_columns = ["avg_vol_last_100"]

# Create a list of features you want to include
features = [col for col in X.columns if col not in exclude_columns]

rf_max_depth_5 = RandomForestClassifier(max_depth=5, random_state=42)
dt_max_depth_5 = DecisionTreeClassifier(max_depth=5, random_state=42)
gb_n_estimators_100 = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_learning_rate_01 = GradientBoostingClassifier(learning_rate=0.1, random_state=42)
gb_max_depth_3 = GradientBoostingClassifier(max_depth=3, random_state=42)
rf_min_samples_split_10 = RandomForestClassifier(min_samples_split=10, random_state=42)


# Define the models you want to use
models = [
    ("Random Forest Classifier (max_depth=5)", rf_max_depth_5),
    ("Decision Tree Classifier (Max Depth 5)", dt_max_depth_5),
    ("Gradient Boosting Classifier", GradientBoostingClassifier(random_state=42)),
    ("Gradient Boosting Classifier (n_estimators=100)", gb_n_estimators_100),
    ("Gradient Boosting Classifier (learning_rate=0.1)", gb_learning_rate_01),
    ("Gradient Boosting Classifier (max_depth=3)", gb_max_depth_3),
    ("Multinomial Naive Bayes", MultinomialNB()),
    ("Random Forest Classifier (min_samples_split=10)", rf_min_samples_split_10),
]

results = []

# Run each model evaluation in its own thread
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(evaluate_model, name, model) for name, model in models]

    for future in futures:
        result = future.result()
        print("Model:", result["Model"])
        print("Best score:", result["Best Score"])
        print("Best features:", result["Best Features"])
        print()
        results.append(result)

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("model_evaluation_results.csv", index=False)
