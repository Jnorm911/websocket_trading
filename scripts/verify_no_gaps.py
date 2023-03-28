import pandas as pd
from datetime import timedelta

# Read the combined CSV file
combined_data = pd.read_csv("kc_btc_1min_combined.csv")

# Convert the 'time' column to datetime
combined_data["time"] = pd.to_datetime(combined_data["time"])

# Sort the DataFrame by the 'time' column
combined_data = combined_data.sort_values(by="time")

# Remove duplicate rows
combined_data = combined_data.drop_duplicates(subset="time", keep="first").reset_index(
    drop=True
)

# Initialize a variable to keep track of gaps
gaps_found = 0

# Iterate through the rows and check for gaps
for i in range(1, len(combined_data)):
    current_time = combined_data.iloc[i]["time"]
    previous_time = combined_data.iloc[i - 1]["time"]
    time_difference = current_time - previous_time

    if time_difference != timedelta(minutes=1):
        print(f"Gap detected between {previous_time} and {current_time}.")
        gaps_found += 1

if gaps_found == 0:
    print("No gaps found.")
else:
    print(f"Found {gaps_found} gap(s) in the data.")
