import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet


def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def arima_model(train_data, order=(1, 1, 1)):
    model = ARIMA(train_data["close"], order=order)
    model_fit = model.fit()
    return model_fit


def create_target_from_arima_predictions(order=(1, 1, 1), look_ahead=1):
    model_fit = arima_model(train_data, order)

    # Make predictions and calculate the binary target
    y_pred = model_fit.predict(
        start=len(train_data), end=len(train_data) + len(test_data) - 1, typ="levels"
    )
    binary_target = np.where(y_pred.shift(-look_ahead) > y_pred, 1, 0)[
        :-look_ahead
    ]  # Remove last look_ahead elements

    return binary_target


def exp_smoothing_model(train_data, seasonal_periods=12, trend="add", seasonal="add"):
    model = ExponentialSmoothing(
        train_data["close"],
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal,
    )
    model_fit = model.fit()
    return model_fit


def lstm_model(train_data, look_back=3, epochs=1):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    X_train, y_train = create_dataset(
        train_data["close"].values.reshape(-1, 1), look_back
    )

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    # Create and fit the LSTM model
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=2)

    return model


def prophet_model(train_data):
    import pandas as pd
    from prophet import Prophet

    # Prepare data for Prophet
    train_df = pd.DataFrame({"ds": train_data.index, "y": train_data["close"]})

    # Fit the prophet model
    model = Prophet()
    model.fit(train_df)
    return model


def get_model_prediction(model_name, train_data, test_data):
    # Sort the index and add frequency information
    train_data = train_data.sort_index()
    train_data = train_data.asfreq("T")

    # Fill missing values
    train_data = train_data.fillna(method="ffill")  # Forward fill missing values

    if model_name == "arima":
        model_fit = arima_model(train_data)
        y_pred = model_fit.predict(
            start=len(train_data),
            end=len(train_data) + len(test_data) - 1,
            typ="levels",
        )
    elif model_name == "exp_smoothing":
        model_fit = exp_smoothing_model(train_data)
        y_pred = model_fit.predict(
            start=len(train_data), end=len(train_data) + len(test_data) - 1
        )
    elif model_name == "lstm":
        model_fit = lstm_model(train_data)
        X_test, _ = create_dataset(
            test_data["close"].values.reshape(-1, 1), look_back=3
        )
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        y_pred = model_fit.predict(X_test)
        y_pred = np.array(y_pred)  # Convert y_pred to a NumPy array
    elif model_name == "prophet":
        model_fit = prophet_model(train_data)
        future = model_fit.make_future_dataframe(periods=len(test_data))
        forecast = model_fit.predict(future)
        y_pred = forecast.tail(len(test_data))["yhat"].values
    else:
        raise ValueError("Invalid model name")

    return y_pred
