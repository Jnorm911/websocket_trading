import pandas as pd
import os

# Define the directory where the CSV files are located
data_dir = "data/kc/btc/heiken_ashi/with_trade_indicators/"

# Loop through all the possible values of x (1 to 60)
for x in range(1, 61):
    # Load the CSV file for the current value of x
    file_name = f"kc_btc_{x}min_ha_ti.csv"
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path)

    try:
        # Rename the 'unix_time' column to 'time'
        df.rename(columns={"unix_time": "time"}, inplace=True)

        # Move the 'time' column to the first position
        cols = ["time"] + [col for col in df.columns if col != "time"]
        df = df[cols]

        # Save the updated dataframe back to the CSV file
        df.to_csv(file_path, index=False)

        # Print a message to show the progress
        print(f"Processed file: {file_name}")

    except KeyError as e:
        print(f"Error processing file: {file_name}. Missing column: {e}")
