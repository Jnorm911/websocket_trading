import itertools
import threading
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def logistic_regression(X_train, X_val, y_train, y_val, features_combination):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train[list(features_combination)], y_train)
    y_pred = model.predict(X_val[list(features_combination)])
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy


def worker(
    combinations_chunk,
    X_train,
    X_val,
    y_train,
    y_val,
    good_combinations,
    lock,
):
    for combination in combinations_chunk:
        accuracy = logistic_regression(X_train, X_val, y_train, y_val, combination)

        if accuracy > 0.65:
            with lock:
                good_combinations.append((combination, accuracy))


def find_good_combinations(X_train, X_val, y_train, y_val, num_threads=16):
    features = X_train.columns
    num_features = len(features)

    good_combinations = []
    lock = threading.Lock()

    for r in range(1, num_features + 1):
        combinations = list(itertools.combinations(features, r))
        chunk_size = len(combinations) // num_threads

        threads = []
        for i in range(num_threads):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_threads - 1 else len(combinations)
            chunk = combinations[start:end]

            thread = threading.Thread(
                target=worker,
                args=(
                    chunk,
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    good_combinations,
                    lock,
                ),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    return good_combinations


# Load the preprocessed data
i = 10
preprocessed_data_path = f"/home/jnorm/kc_btc_{i}min_ha_ti_pro.csv"
df = pd.read_csv(preprocessed_data_path)

# Define the target variable and remove it from the features list
target_column = "color_change"
time_column = "time"
X = df.drop(columns=[target_column, time_column])
y = df[target_column]

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

print("Training, validation, and test sets created.")
good_combinations = find_good_combinations(X_train, X_val, y_train, y_val)

# Save good combinations to a CSV file
good_combinations_df = pd.DataFrame(
    good_combinations, columns=["combination", "accuracy"]
)
good_combinations_df.to_csv("good_combinations.csv", index=False)

print("Good combinations saved to 'good_combinations.csv'.")
