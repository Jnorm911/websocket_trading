import os
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
    VotingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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

# Bagging
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    random_state=42,
)
# RUN ITERTOOLS ON THIS
voting_clf = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("gnb", GaussianNB()),
    ],
    voting="hard",
)
# AdaBoost
adaboost_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    random_state=42,
)
# Stacking (use a combination of classifiers you already have)
stacking_clf = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("gnb", GaussianNB()),
    ],
    final_estimator=LogisticRegression(random_state=42),
)
# HistGradientBoostingClassifier
histgradient_clf = HistGradientBoostingClassifier(random_state=42)

# Decision Tree Classifier with different maximum depths
dt_max_depth_3 = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_max_depth_5 = DecisionTreeClassifier(max_depth=5, random_state=42)

# Decision Tree Classifier with different minimum samples required to split an internal node
dt_min_samples_split_10 = DecisionTreeClassifier(min_samples_split=10, random_state=42)
dt_min_samples_split_20 = DecisionTreeClassifier(min_samples_split=20, random_state=42)

# Decision Tree Classifier with different splitting criteria
dt_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
dt_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)

rf_n_estimators_100 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_max_depth_5 = RandomForestClassifier(max_depth=5, random_state=42)
rf_min_samples_split_10 = RandomForestClassifier(min_samples_split=10, random_state=42)

gb_n_estimators_100 = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_learning_rate_01 = GradientBoostingClassifier(learning_rate=0.1, random_state=42)
gb_max_depth_3 = GradientBoostingClassifier(max_depth=3, random_state=42)

knn_3_neighbors = KNeighborsClassifier(n_neighbors=3)
knn_7_neighbors = KNeighborsClassifier(n_neighbors=7)
knn_distance_weights = KNeighborsClassifier(weights="distance")

svc_poly_kernel = SVC(kernel="poly", degree=3, random_state=42)
svc_c_05 = SVC(C=0.5, random_state=42)

mlp_hidden_layer_sizes = MLPClassifier(hidden_layer_sizes=(50, 50), random_state=42)
mlp_activation_tanh = MLPClassifier(activation="tanh", random_state=42)
mlp_learning_rate_init = MLPClassifier(learning_rate_init=0.01, random_state=42)


models = [
    # Classification models
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("Random Forest Classifier", RandomForestClassifier(random_state=42)),
    ("Gradient Boosting Classifier", GradientBoostingClassifier(random_state=42)),
    ("Support Vector Classification", SVC(kernel="rbf", random_state=42)),
    ("k-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
    ("Decision Tree Classifier", DecisionTreeClassifier(random_state=42)),
    ("Gaussian Naive Bayes", GaussianNB()),
    ("Multinomial Naive Bayes", MultinomialNB()),
    ("Bernoulli Naive Bayes", BernoulliNB()),
    ("Multilayer Perceptron Classifier", MLPClassifier(random_state=42)),
    ("Stochastic Gradient Descent Classifier", SGDClassifier(random_state=42)),
    ("Extra Trees Classifier", ExtraTreesClassifier(random_state=42)),
    ("Bagging", bagging_clf),
    ("Voting", voting_clf),
    ("AdaBoost", adaboost_clf),
    ("Stacking", stacking_clf),
    ("HistGradientBoosting", histgradient_clf),
    ("Decision Tree Classifier (Max Depth 5)", dt_max_depth_5),
    ("Decision Tree Classifier (Min Samples Split 10)", dt_min_samples_split_10),
    ("Decision Tree Classifier (Min Samples Split 20)", dt_min_samples_split_20),
    ("Decision Tree Classifier (Gini)", dt_gini),
    ("Decision Tree Classifier (Entropy)", dt_entropy),
    ("Random Forest Classifier (n_estimators=100)", rf_n_estimators_100),
    ("Random Forest Classifier (max_depth=5)", rf_max_depth_5),
    ("Random Forest Classifier (min_samples_split=10)", rf_min_samples_split_10),
    ("Gradient Boosting Classifier (n_estimators=100)", gb_n_estimators_100),
    ("Gradient Boosting Classifier (learning_rate=0.1)", gb_learning_rate_01),
    ("Gradient Boosting Classifier (max_depth=3)", gb_max_depth_3),
    ("K-Nearest Neighbors Classifier (n_neighbors=3)", knn_3_neighbors),
    ("K-Nearest Neighbors Classifier (n_neighbors=7)", knn_7_neighbors),
    ("K-Nearest Neighbors Classifier (weights='distance')", knn_distance_weights),
    ("Support Vector Classifier (polynomial kernel, degree=3)", svc_poly_kernel),
    ("Support Vector Classifier (C=0.5)", svc_c_05),
    (
        "Multi-Layer Perceptron Classifier (hidden_layer_sizes=(50, 50))",
        mlp_hidden_layer_sizes,
    ),
    ("Multi-Layer Perceptron Classifier (activation='tanh')", mlp_activation_tanh),
    (
        "Multi-Layer Perceptron Classifier (learning_rate_init=0.01)",
        mlp_learning_rate_init,
    )
    # Custom implementation models
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
        model_results.append((name, cv_scores_mean, ""))

        # Print the current model's cross-validation score
        print(f"{name} cross-validation score: {cv_scores_mean}\n")

    except Exception as e:
        print(f"Error occurred while training {name}: {e}\n")
        continue

# Display the results
print("Cross-validation scores for all models:")
for name, score, _ in model_results:  # Add an underscore to ignore the "Type" value
    print(f"{name}: {score}")

# Choose the best model based on the cross-validation scores
best_model_name, best_model_score, _ = max(
    model_results, key=lambda x: x[1]
)  # Add an underscore to ignore the "Type" value

print(f"\nBest model: {best_model_name} with score: {best_model_score}")

# Train the best model on the combined training and validation set
best_model = [model for name, model in models if name == best_model_name][0]
best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

# Evaluate the best model on the test set
test_score = best_model.score(X_test, y_test)
print(f"Test accuracy of the best model ({best_model_name}): {test_score}")

# Convert the model_results list to a DataFrame
new_results_df = pd.DataFrame(model_results, columns=["Model", "Score", "Type"])

# Ask the user for input
user_input = input("Please enter the type: ")

# Add a new column "Type" filled with the user's input
new_results_df["Type"] = user_input

# If the CSV file exists, read the existing data, concatenate the new results, and save it
if os.path.exists("model_results.csv"):
    existing_results_df = pd.read_csv("model_results.csv")
    combined_results_df = pd.concat(
        [existing_results_df, new_results_df], ignore_index=True
    )
    combined_results_df.to_csv("model_results.csv", index=False)
else:
    # If the CSV file doesn't exist, save the new results to the CSV file
    new_results_df.to_csv("model_results.csv", index=False)

# Display the DataFrame in the console
print(new_results_df)
