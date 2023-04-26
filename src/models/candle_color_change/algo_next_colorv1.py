import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Perform k-fold cross-validation
def evaluate_fold(train_index, test_index, clf, metrics):
    X_train_fold, X_test_fold = X_train.loc[train_index], X_train.loc[test_index]
    y_train_fold, y_test_fold = y_train.loc[train_index], y_train.loc[test_index]

    # Train the classifier
    clf.fit(X_train_fold, y_train_fold)

    # Predict the target variable
    y_pred = clf.predict(X_test_fold)

    # Calculate the evaluation metrics
    fold_scores = {}
    for metric, score_func in metrics.items():
        score = score_func(y_test_fold, y_pred)
        fold_scores[metric] = score

    return fold_scores


file_path = rf"data\kc\btc\heiken_ashi\with_trade_indicators\standard\kline\kc_btc_12min_ha_ti_pro.csv"
data = pd.read_csv(file_path)

# Select only the feature-fitted columns
selected_columns = [
    "ATR",
    "BBB_5_2.0",
    "BBP_5_2.0",
    "CCI",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
    "ROC",
    "RSI",
    "S3",
    "STOCHd_14_3_3",
    "STOCHk_14_3_3",
    "avg_vol_last_100",
    "time",
    "turnover",
    "volume",
    "open",
    "high",
    "low",
    "close",
    "color_change",
]
data = data[selected_columns]

target = "color_change"
X_train = data.drop(columns=[target])
y_train = data[target]

# Initialize the metrics
metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1_score": f1_score,
    "roc_auc": roc_auc_score,
}

# Initialize the results DataFrame
results = pd.DataFrame(columns=["Classifier"] + list(metrics.keys()))

# Define the classifiers
classifiers = [
    (
        "Logistic Regression",
        LogisticRegression(random_state=42, solver="liblinear", max_iter=10000),
    ),
    ("Naive Bayes", GaussianNB()),
    ("Support Vector Machines", SVC(probability=True)),
]

# Initialize k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Evaluate the classifiers
for name, clf in tqdm(classifiers, desc="Classifiers"):
    print(f"Classifier: {name}")

    # Initialize lists to store the evaluation metrics
    scores = {metric: [] for metric in metrics.keys()}

    fold_scores_list = Parallel(n_jobs=-1)(
        delayed(evaluate_fold)(train_index, test_index, clf, metrics)
        for train_index, test_index in kf.split(X_train, y_train)
    )

    # Aggregate scores from each fold
    for fold_scores in fold_scores_list:
        for metric in metrics.keys():
            scores[metric].append(fold_scores[metric])

    # Calculate the average score for each metric
    avg_scores = {
        metric: np.mean(scores_list) for metric, scores_list in scores.items()
    }

    # Append the results to the results DataFrame
    results = pd.concat(
        [results, pd.DataFrame({"Classifier": name, **avg_scores}, index=[0])],
        ignore_index=True,
    )
    # Print the results
    print("Average scores:")
    for metric, avg_score in avg_scores.items():
        print(f"{metric}: {avg_score:.4f}")
    print("\n")
