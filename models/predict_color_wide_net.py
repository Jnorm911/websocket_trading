import itertools
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

file_path = "C:/Users/jnorm/Projects/websocket_trading/data/kc/btc/heiken_ashi/with_trade_indicators/kc_btc_59min_ha_ti.csv"

df = pd.read_csv(file_path)


# Preprocess the data
df = df.dropna()
df["color"] = df["color"].map({"red": 0, "green": 1})

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

tech_indicators = [
    "avg_vol_last_100",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
    "PP",
    "R1",
    "S1",
    "R2",
    "S2",
    "R3",
    "S3",
    "BBL_5_2.0",
    "BBM_5_2.0",
    "BBU_5_2.0",
    "BBB_5_2.0",
    "BBP_5_2.0",
    "RSI",
]

# Dictionary to store the accuracy for each combination
accuracies = {}

# Loop through all possible combination lengths
for r in range(1, len(tech_indicators) + 1):
    combinations = itertools.combinations(tech_indicators, r)

    for combo in combinations:
        X_train = train_df[list(combo)]
        y_train = train_df["color"]
        X_test = test_df[list(combo)]
        y_test = test_df["color"]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        accuracies[combo] = accuracy

        # Print the current combination and its accuracy
        print(f"Combination: {combo}, Accuracy: {accuracy}")

# Print the accuracies for each combination
print(accuracies)

# Convert the accuracies dictionary into a DataFrame
accuracy_df = pd.DataFrame(
    list(accuracies.items()), columns=["Combination", "Accuracy"]
)

# Add a column for the number of indicators in each combination
accuracy_df["Num_Indicators"] = accuracy_df["Combination"].apply(len)

# Add a column for the indicator names as a string
accuracy_df["Indicator_Names"] = accuracy_df["Combination"].apply(
    lambda x: ", ".join(x)
)

# Save the accuracy_df to a CSV file
accuracy_df.to_csv("accuracy.csv", index=False)
