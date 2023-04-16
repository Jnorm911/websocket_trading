import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import csv
import gc
import os


def evaluate_combination(columns_combination):
    X_train_subset = X_train[list(columns_combination)]
    X_val_subset = X_val[list(columns_combination)]

    # Create a random forest model
    model = RandomForestClassifier(
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=5,
        n_estimators=200,
        n_jobs=-1,
    )
    model.fit(X_train_subset, y_train)

    # Calculatethe accuracy on the validation set
    y_val_pred = model.predict(X_val_subset)
    accuracy = accuracy_score(y_val, y_val_pred)

    return accuracy, columns_combination


def main(X_train, X_val, y_train, y_val):
    print("Starting the main function")

    # Open a new CSV file to store the results
    with open("results.csv", mode="w", newline="") as results_file:
        results_writer = csv.writer(results_file)
        results_writer.writerow(["Column Combination", "Validation Accuracy"])

        with ProcessPoolExecutor(os.cpu_count()) as executor:
            futures = []

            for r in range(1, len(X_train.columns) + 1):
                for columns_combination in itertools.combinations(X_train.columns, r):
                    future = executor.submit(
                        evaluate_combination, columns_combination, results_writer
                    )
                    futures.append(future)

                # Clear memory for the current iteration
                del columns_combination
                gc.collect()

            for completed_future in as_completed(futures):
                accuracy, columns_combination = completed_future.result()
                if accuracy >= 0.65:
                    results_writer.writerow((str(columns_combination), accuracy))

                # Remove the completed future from the list
                futures.remove(completed_future)
                del completed_future
                gc.collect()

    # Clear memory after the loop
    del X_train, X_val, y_train, y_val
    gc.collect()


if __name__ == "__main__":
    print("Loading the dataset")

    # Load the preprocessed data for a specific candle length (replace 'i' with the desired value)
    i = 59  # or any other value between 1 and 60
    # preprocessed_data_path = f"/home/jnorm/kc_btc_{i}min_ha_ti_pro.csv"
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

    main(X_train, X_val, y_train, y_val)
