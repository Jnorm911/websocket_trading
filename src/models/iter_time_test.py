import time
from itertools import combinations
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# Replace these with the best configurations you found earlier
best_configs = {
    "LogisticRegression": {...},
    "DecisionTreeClassifier": {...},
    "GaussianNB": {...},
    "LinearSVC": {...},
    "SGDClassifier": {...},
}

# Instantiate the models with their best configurations
models = {
    "LogisticRegression": LogisticRegression(
        **best_configs["LogisticRegression"]["config"]["params"]
    ),
    "DecisionTreeClassifier": DecisionTreeClassifier(
        **best_configs["DecisionTreeClassifier"]["config"]["params"]
    ),
    "GaussianNB": GaussianNB(**best_configs["GaussianNB"]["config"]["params"]),
    "LinearSVC": LinearSVC(**best_configs["LinearSVC"]["config"]["params"]),
    "SGDClassifier": SGDClassifier(**best_configs["SGDClassifier"]["config"]["params"]),
}

# Time limit for finding combinations
time_limit = 60  # seconds

# Loop through each model and count the number of combinations found within the time limit
combinations_found = {}
for model_name, model in models.items():
    start_time = time.time()
    combinations_count = 0

    while time.time() - start_time < time_limit:
        for feature_combination in combinations(
            range(10), 2
        ):  # Adjust range and size of combinations as needed
            combinations_count += 1

    combinations_found[model_name] = combinations_count

# Print the number of combinations found for each model
for model_name, count in combinations_found.items():
    print(f"Model: {model_name}")
    print(f"  Combinations found in {time_limit} seconds: {count}\n")
