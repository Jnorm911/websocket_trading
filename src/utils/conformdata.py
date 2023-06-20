import pandas as pd
import os

# Iterate over all files
for i in range(1, 61):
    file_path = f"data/kc/btc/raw/kc_btc_{i}min.csv"

    if os.path.exists(file_path):
        # Load the data
        df = pd.read_csv(file_path)

        # Rename the column
        df.rename(columns={"unix_time": "time"}, inplace=True)

        # Save the data
        df.to_csv(file_path, index=False)
    else:
        print(f"File not found: {file_path}")
