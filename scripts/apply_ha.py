import os
import pandas as pd
import pandas_ta as ta


# Function to add Heikin Ashi columns to a DataFrame
def add_heikin_ashi_columns(df):
    heikin_ashi = ta.ha(df["open"], df["high"], df["low"], df["close"])
    df["open"] = heikin_ashi["HA_open"]
    df["high"] = heikin_ashi["HA_high"]
    df["low"] = heikin_ashi["HA_low"]
    df["close"] = heikin_ashi["HA_close"]
    return df


# Directory containing the CSV files
csv_directory = "data/kc/btc/raw"

# Directory to save the updated CSV files
output_directory = "data/kc/btc/heiken_ashi"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through each file in the directory
for file_name in os.listdir(csv_directory):
    if file_name.startswith("kc_btc_") and file_name.endswith(".csv"):
        file_path = os.path.join(csv_directory, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Add the Heikin Ashi columns
        df = add_heikin_ashi_columns(df)

        # Create a new file name with '_ha' appended before the file extension
        new_file_name = file_name[:-4] + "_ha.csv"
        new_file_path = os.path.join(output_directory, new_file_name)

        # Save the updated DataFrame to the new file
        df.to_csv(new_file_path, index=False)
        print(
            f"Updated {file_name} with Heikin Ashi columns and saved as {new_file_name} in {output_directory}"
        )
