import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearnex import patch_sklearn
import ray
from ray import tune

# Load and preprocess your dataset here

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Patch scikit-learn with scikit-learn-intelex
patch_sklearn()
ray.init()


# Custom training function
def train_model(config):
    model = None
    if config["model"] == "LogisticRegression":
        model = LogisticRegression(**config["params"])
    elif config["model"] == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(**config["params"])
    elif config["model"] == "GaussianNB":
        model = GaussianNB(**config["params"])
    elif config["model"] == "LinearSVC":
        model = LinearSVC(**config["params"])
    elif config["model"] == "SGDClassifier":
        model = SGDClassifier(**config["params"])

    all_feature_combinations = [
        comb for r in range(1, len(features) + 1) for comb in combinations(features, r)
    ]

    best_score = -np.inf
    for feature_combination in all_feature_combinations:
        X_train_subset = X_train[list(feature_combination)]
        score = np.mean(cross_val_score(model, X_train_subset, y_train, cv=5))
        if score > best_score:
            best_score = score
            best_combination = feature_combination

    tune.report(mean_score=best_score, best_combination=best_combination)


# Hyperparameter search space and model configurations
def search_space_for_model(model_name):
    if model_name == "LogisticRegression":
        return {
            "solver": tune.choice(["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
            "C": tune.loguniform(1e-5, 100),
            "penalty": tune.choice(["l1", "l2", "elasticnet", "none"]),
            "max_iter": tune.randint(100, 1000),
        }
    if model_name == "DecisionTreeClassifier":
        return {
            "criterion": tune.choice(["gini", "entropy"]),
            "max_depth": tune.randint(1, 100),
            "min_samples_split": tune.uniform(0.1, 1),
            "min_samples_leaf": tune.uniform(0.1, 0.5),
        }
    if model_name == "GaussianNB":
        return {"var_smoothing": tune.loguniform(1e-10, 1e-2)}
    if model_name == "LinearSvc":
        return {
            "C": tune.loguniform(1e-5, 100),
            "penalty": tune.choice(["l1", "l2"]),
            "loss": tune.choice(["hinge", "squared_hinge"]),
            "max_iter": tune.randint(100, 1000),
        }
    if model_name == "SGDClassifier":
        return {
            "loss": tune.choice(
                ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
            ),
            "penalty": tune.choice(["l1", "l2", "elasticnet"]),
            "alpha": tune.loguniform(1e-5, 1),
            "max_iter": tune.randint(100, 1000),
        }


# Iterate through the models
model_names = [
    "LogisticRegression",
    "DecisionTreeClassifier",
    "GaussianNB",
    "LinearSVC",
    "SGDClassifier",
]
best_configs = {}

for model_name in model_names:
    # Set up the model-specific search space
    search_space = {"model": model_name, "params": search_space_for_model(model_name)}

    # Run Ray Tune for the current model
    analysis = tune.run(
        train_model,
        resources_per_trial={"cpu": 1, "gpu": 0},
        config=search_space,
        num_samples=100,
        local_dir="tune_results",
        name=f"tune_hyperparameters_{model_name}",
        metric="mean_score",
        mode="max",
        stop={"training_iteration": 10},
        progress_reporter=tune.JupyterNotebookReporter(overwrite=True),
        verbose=1,
    )

    # Get the best trial for the current model
    best_trial = analysis.get_best_trial("mean_score", "max", "last")
    best_config = best_trial.config
    best_score = best_trial.last_result["mean_score"]
    best_combination = best_trial.last_result["best_combination"]

    # Store the best configuration for the current model
    best_configs[model_name] = {
        "config": best_config,
        "score": best_score,
        "combination": best_combination,
    }

# Print the best configurations for each model
for model_name, config_info in best_configs.items():
    print(f"Model: {model_name}")
    print(f"  Best configuration: {config_info['config']}")
    print(f"  Best feature combination: {config_info['combination']}")
    print(f"  Best score: {config_info['score']}\n")
