import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# Convert an array of values into a dataset matrix
def create_dataset(X, Y, look_back=1):
    dataX, dataY = [], []
    for i in range(len(X) - look_back - 1):
        a = X[i : (i + look_back), :]
        dataX.append(a)
        dataY.append(Y[i + look_back])
    return np.array(dataX), np.array(dataY)


data_path = "data/kc/btc/heiken_ashi/with_trade_indicators/raw/kc_btc_60min_ha_ti.csv"
df = pd.read_csv(data_path)

# Create color_change feature
df["color_change"] = df["color"].ne(df["color"].shift()).astype(int)

# Convert color to binary: 'green' as 1, 'red' as 0
df["color"] = df["color"].map({"green": 1, "red": 0})

# Drop the time column
df.drop(["time"], axis=1, inplace=True)

# Feature Engineering

# Lagged features
for i in range(1, 4):  # Creating 3 lags
    df[f"lag{i}_MACD_12_26_9"] = df["MACD_12_26_9"].shift(i)

# Volatility
df["volatility"] = df["close"].rolling(window=5).std()  # adjust window as needed

# Remove any rows with missing values
df = df.dropna()

# Define the features to scale
features_to_scale = df.columns.drop(
    [
        "color",
        "color_change",
        # "RSI_5",
        # "RSI_10",
        # "RSI_14",
    ]
)

# Scale the selected features
scaler = MinMaxScaler(feature_range=(0, 1))
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Feature Selection
X = df.drop("color_change", axis=1)
y = df["color_change"]

estimator = RandomForestClassifier()
selector = RFE(
    estimator, n_features_to_select=15, step=1
)  # adjust parameters as needed
X = selector.fit_transform(X, y)

# Class Imbalance
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

# Scale X
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# Convert y to categorical
y = to_categorical(y)

# Reshape into X=t and Y=t+1
look_back = 3
X, y = create_dataset(X, y, look_back)

# Split into train and test sets
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
trainX, testX = X[0:train_size, :], X[train_size : len(X), :]
trainY, testY = y[0:train_size, :], y[train_size : len(y), :]

# Initialize LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(2, activation="softmax"))

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# Train the model
model.fit(trainX, trainY, epochs=50, batch_size=32, verbose=2)
