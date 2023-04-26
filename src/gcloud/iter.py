import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from io import StringIO
from tqdm import tqdm
from joblib import Parallel, delayed


# Load and preprocess the dataset
def load_data():
    file_path = f"/home/jnorm/kline/kc_btc_12min_ha_ti_pro.csv"
    # file_path = rf"./standard_kline.csv"
    data = pd.read_csv(file_path)
    target = "color_change"
    X = data.drop(columns=[target])
    y = data[target]
    return X, y


def generate_combinations(features, n):
    return list(itertools.combinations(features, n))


def train_test_combination(combination, X_train, X_test, y_train, y_test):
    try:
        X_train_comb = X_train[list(combination)]
        X_test_comb = X_test[list(combination)]

        clf = GaussianNB()
        clf.fit(X_train_comb, y_train)
        y_pred = clf.predict(X_test_comb)

        accuracy = accuracy_score(y_test, y_pred)
        return combination, accuracy
    except Exception as e:
        print(f"Error processing combination {combination}: {e}")
        return combination, None


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, random_state=42
    )

    features = [
        "MACDs_12_26_9",
        "PP",
        "R1",
        "S1",
        "R2",
        "S2",
        "R3",
        "S3",
        "BBP_5_2.0",
        "RSI",
        "SMA_5",
        "SMA_10",
        "EMA_5",
        "EMA_10",
        "STOCHd_14_3_3",
        "ATR",
        "ROC",
        "CCI",
    ]
    min_combination_size = 6
    max_combination_size = 24

    for n in tqdm(
        range(min_combination_size, max_combination_size + 1),
        total=max_combination_size - min_combination_size + 1,
    ):
        combinations = generate_combinations(features, n)

        results = Parallel(n_jobs=4)(
            delayed(train_test_combination)(
                combination, X_train, X_test, y_train, y_test
            )
            for combination in combinations
        )

        # Create a DataFrame to store the results for the current combination size
        df_results = pd.DataFrame(columns=["combination", "accuracy"])

        for combination, accuracy in results:
            if accuracy is not None and accuracy > 0.5:
                df_results = df_results.append(
                    {"combination": "-".join(combination), "accuracy": accuracy},
                    ignore_index=True,
                )

        # Save the results for the current combination size to a CSV file
        df_results.to_csv(f"combinations_accuracy_{n}.csv", index=False)


if __name__ == "__main__":
    main()
