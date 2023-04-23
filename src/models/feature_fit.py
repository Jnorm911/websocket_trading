import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split as dask_train_test_split
from dask_ml.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif

# Load the preprocessed data
file_path = rf"data\kc\btc\heiken_ashi\with_trade_indicators\standard\kline\kc_btc_12min_ha_ti_pro.csv"
df = pd.read_csv(file_path)

ddf = dd.from_pandas(df, npartitions=2)

# Define the target variable and remove it from the features list
target_column = "color_change"
X = ddf.drop(columns=[target_column])
y = ddf[target_column]

X_train, X_test, y_train, y_test = dask_train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=False
)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create classifiers
lda = LDA()
lr = LogisticRegression(max_iter=1000)
nb = GaussianNB()
svm = SVC()

# Feature selection and evaluation
classifiers = [
    ("LDA", lda),
    ("Logistic Regression", lr),
    ("Naive Bayes", nb),
    ("Support Vector Machines", svm),
]

results = []
cv = StratifiedKFold(n_splits=10)

for name, classifier in classifiers:
    print(f"Feature Selection and Evaluation for {name}:")
    best_score = -np.inf
    best_k = -1
    best_features = None

    for k in range(1, X_train.shape[1] + 1):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train.compute(), y_train.compute())

        scores = cross_val_score(classifier, X_train_selected, y_train.compute(), cv=cv)
        score = np.mean(scores)

        print(f"  {k} features: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_features = X.columns[selector.get_support()]

    results.append([name, best_k, best_score, ",".join(best_features)])

# Save results to CSV
results_df = pd.DataFrame(
    results, columns=["Classifier", "Best K", "Best Accuracy", "Best Features"]
)
results_df.to_csv("feature_selection_results.csv", index=False)

print("Results saved to feature_selection_results.csv")
