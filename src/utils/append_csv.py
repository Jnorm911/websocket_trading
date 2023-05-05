import pandas as pd

# Read the CSV files
file1 = "linear_minmax_results.csv"
file2 = "linear_minmax_results3.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Concatenate the dataframes
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the concatenated dataframe to a new CSV file
output_file = "combined_linear_minmax_results.csv"
combined_df.to_csv(output_file, index=False)
