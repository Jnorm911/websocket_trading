import pandas as pd
from sklearn.model_selection import train_test_split

## TTV Test, Train, & Validation ##

# Load the preprocessed data for a specific candle length (replace 'i' with the desired value)
i = 59  # or any other value between 1 and 60
preprocessed_data_path = f"data/kc/btc/heiken_ashi/with_trade_indicators/processed/kc_btc_{i}min_ha_ti_pro.csv"
df = pd.read_csv(preprocessed_data_path)

# Predicting the next candle color #
# Define your target variable (column) and remove it from the features list
target_column = "color_green"
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
