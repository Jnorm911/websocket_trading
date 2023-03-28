import pandas as pd
import os

num_files = 85
output_filename = "kc_btc_1min_combined.csv"

# Initialize an empty DataFrame
full_data = pd.DataFrame()

for i in range(num_files):
    input_filename = f"kc_btc_1min_day_{i}.csv"

    # Check if the file exists
    if os.path.isfile(input_filename):
        # Read the data from the CSV file
        data = pd.read_csv(input_filename)

        # Append the data to the full_data DataFrame
        full_data = pd.concat([full_data, data], ignore_index=True)

# Save the combined data to a new CSV file
full_data.to_csv(output_filename, index=False)
print(f"Data combined and saved to {output_filename}")
