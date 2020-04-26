# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/aerfio/e27e96c3e2dbc616a807b5c709c84de5/main.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import datetime
import os
import time
import tensorflow as tf

USING_TPU = True  # @param {type:"boolean"}
# bucket = 'tensorboard-logs-master-thesis' #@param {type:"string"}


if USING_TPU:
    assert 'COLAB_TPU_ADDR' in os.environ, 'Missing TPU; did you request a TPU in Notebook Settings?'

# MODEL_DIR = 'gs://{}/{}'.format(bucket, time.strftime('tpuestimator-lstm/%Y-%m-%d-%H-%M-%S'))
# LOG_DIR = 'gs://{}/{}'.format(bucket, time.strftime('tpuestimator-lstm-logs/%Y-%m-%d-%H-%M-%S'))
# print('Using model dir: {}'.format(MODEL_DIR))

# from google.colab import auth
# auth.authenticate_user()

try:
    device_name = os.environ["COLAB_TPU_ADDR"]
    TPU_ADDRESS = "grpc://" + device_name
    print("Found TPU at: {}".format(TPU_ADDRESS))
except KeyError:
    print("TPU not found, probably using GPU")

if USING_TPU:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=TPU_ADDRESS)
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

dataset_location = "https://raw.githubusercontent.com/aerfio/master-theis-datasets/master/datasets/eurusd-m15-2018/EURUSD.csv"

df = pd.read_csv(dataset_location)

print(df.head())

print(df.count())

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

# Add additional features
df['momentum'] = df['volume'] * (df['open'] - df['close'])
df['avg_price'] = (df['low'] + df['high']) / 2
# df['range'] = df['high'] - df['low']
df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close']) / 4
df['oc_diff'] = df['open'] - df['close']


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


def create_model():
    model = Sequential()
    model.add(
        LSTM(60, input_shape=(X.shape[1], X.shape[2]),
             return_sequences=True))

    model.add(LSTM(30, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(15, return_sequences=False))
    model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(6, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam', metrics=['mae', 'mse'])

    print(model.summary())
    return model


if USING_TPU:
    with strategy.scope():
        model = create_model()
else:
    model = create_model()

print(f'Using {"TPU" if USING_TPU else "GPU"}')
history = model.fit(trainX, trainY, validation_split=0.2, epochs=8,
                    batch_size=64, verbose=1)

score = model.evaluate(testX, testY, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
