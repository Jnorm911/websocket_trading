import pandas as pd

x = []
x.append(1)
y = x
x.append(2)
y.append(3)

x = None

print(x)
print(y)

# # Set display options to show all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Load the data from the CSV file
df = pd.read_csv("src/binary_classification/lstm/model_accuracy_results.csv")

# Sort the DataFrame by the 'accuracy' column in ascending order
df_sorted = df.sort_values("accuracy", ascending=True)

# Print the sorted DataFrame
print(df_sorted)
