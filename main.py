# import os  # noqa
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"  # noqa

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
# import keras.backend as K


# import seaborn as sns

dataset_location = "./datasets/eurusd-m15-2018/EURUSD.csv"

df = pd.read_csv(dataset_location)

print(df.head())

df.rename(columns={
    'Open': 'open', 'Close': 'close',
    'High': 'high', 'Low': 'low',
    'Close': 'close', 'Volume': 'volume',
    "Date": "date", 'Timestamp': 'timestamp', }, inplace=True)


df["timestamp"] = df["date"].astype(str) + " " + df["timestamp"]
df.drop("date", 1, inplace=True)
df.rename(columns={'Time': 'timestamp', 'Open': 'open', 'Close': 'close',
                   'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
df.set_index('timestamp', inplace=True)
df = df.astype(float)

# # Add additional features
# df['momentum'] = df['volume'] * (df['open'] - df['close'])
# df['avg_price'] = (df['low'] + df['high']) / 2
# # df['range'] = df['high'] - df['low']
# df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close']) / 4
# df['oc_diff'] = df['open'] - df['close']

print(df.head())
print(df.count())


def create_dataset(dataset, look_back=20):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# Scale and create datasets
target_index = df.columns.tolist().index('close')
high_index = df.columns.tolist().index('high')
low_index = df.columns.tolist().index('low')
dataset = df.values.astype('float32')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Create y_scaler to inverse it later
y_scaler = MinMaxScaler(feature_range=(0, 1))
t_y = df['close'].values.astype('float32')
t_y = np.reshape(t_y, (-1, 1))
y_scaler = y_scaler.fit(t_y)

X, y = create_dataset(dataset, look_back=50)
y = y[:, target_index]

train_size = int(len(X) * 0.99)
trainX = X[:train_size]
trainY = y[:train_size]
testX = X[train_size:]
testY = y[train_size:]

print(trainX[0][0])

model = Sequential()
model.add(
    Bidirectional(LSTM(30, input_shape=(X.shape[1], X.shape[2]),
                       return_sequences=True),
                  merge_mode='sum',
                  weights=None,
                  input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(10, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(4, return_sequences=False))
model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam', metrics=['mae', 'mse'])
print(model.summary())

history = model.fit(trainX, trainY, validation_split=0.2, epochs=1,
                    batch_size=32, verbose=1)

score = model.evaluate(testX, testY, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
