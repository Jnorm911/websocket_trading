import itertools
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import multiprocessing
import time


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
    best_results,
    start_time,
    time_limit_reached,
):
    print(f"Worker started with chunk size: {len(combinations_chunk)}")
    for combination in combinations_chunk:
        if time.time() - start_time >= 60:
            with time_limit_reached.get_lock():
                time_limit_reached.value = True
            break

        if time_limit_reached.value:
            break

        accuracy = logistic_regression(X_train, X_val, y_train, y_val, combination)

        with best_results["lock"]:
            if accuracy > best_results["best_accuracy"]:
                best_results["best_accuracy"] = accuracy
                best_results["best_combination"] = combination


def find_best_combination(X_train, X_val, y_train, y_val, num_processes=24):
    features = X_train.columns
    num_features = len(features)

    best_results = {
        "best_accuracy": 0,
        "best_combination": None,
        "lock": multiprocessing.Lock(),
    }

    time_limit_reached = multiprocessing.Value("b", False)
    start_time = time.time()

    for r in range(1, num_features + 1):
        combinations = list(itertools.combinations(features, r))
        chunk_size = len(combinations) // num_processes

        processes = []
        for i in range(num_processes):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_processes - 1 else len(combinations)
            chunk = combinations[start:end]

            process = multiprocessing.Process(
                target=worker,
                args=(
                    chunk,
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    best_results,
                    start_time,
                    time_limit_reached,
                ),
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        if time_limit_reached.value:
            break

    combo_count = sum([len(chunk) for chunk in combinations])
    print(f"Number of combinations tested in 1 minute: {combo_count}")
    return best_results["best_combination"], best_results["best_accuracy"]


def main():
    # Load the preprocessed data
    i = 59
    preprocessed_data_path = f"data/kc/btc/heiken_ashi/with_trade_indicators/processed/kc_btc_{i}min_ha_ti_pro.csv"
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
    best_combination, best_accuracy = find_best_combination(
        X_train, X_val, y_train, y_val
    )
    print("Best combination of features:", best_combination)
    print("Best accuracy:", best_accuracy)


if __name__ == "__main__":
    main()
