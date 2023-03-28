import pandas as pd

# Read the CSV file into a DataFrame
data = pd.read_csv("kc_btc_1min_combined.csv")

# Convert the 'time' column to datetime
data["time"] = pd.to_datetime(data["time"])

# Sort the DataFrame by the 'time' column in ascending order
sorted_data = data.sort_values(by="time")

# Save the sorted DataFrame to a new CSV file
sorted_data.to_csv("kc_btc_1min_sorted.csv", index=False)
