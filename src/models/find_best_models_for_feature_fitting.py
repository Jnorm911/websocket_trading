import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from dask_ml.model_selection import train_test_split as dask_train_test_split
import dask.dataframe as dd

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

# Create classifiers
lr = LogisticRegression(max_iter=1000)
gnb = GaussianNB()
knn = KNeighborsClassifier()

# Create PCA and LDA transformers
pca = PCA()
lda = LDA()

# Create pipelines for PCA and LDA with each classifier
pipelines = [
    (
        "PCA",
        [
            ("Logistic Regression", Pipeline([("pca", pca), ("classifier", lr)])),
            ("Naive Bayes", Pipeline([("pca", pca), ("classifier", gnb)])),
            ("k-Nearest Neighbors", Pipeline([("pca", pca), ("classifier", knn)])),
        ],
    ),
    (
        "LDA",
        [
            ("Logistic Regression", Pipeline([("lda", lda), ("classifier", lr)])),
            ("Naive Bayes", Pipeline([("lda", lda), ("classifier", gnb)])),
            ("k-Nearest Neighbors", Pipeline([("lda", lda), ("classifier", knn)])),
        ],
    ),
]

# Create a TimeSeriesSplit cross-validator
# cv = TimeSeriesSplit(n_splits=10)

# Evaluate each pipeline using cross_val_score
for transform_name, pipeline_list in pipelines:
    print(f"{transform_name}:")
    for name, pipeline in pipeline_list:
        scores = cross_val_score(pipeline, X_train.compute(), y_train.compute())
        mean, std = scores.mean(), scores.std()
        print(f"  {name}: {mean:.4f} +/- {std:.4f}")
