#!/usr/bin/env python
# coding: utf-8

# In[100]:


# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense 
from keras.optimizers import Adam
# from keras.callbacks import TensorBoard
import time
# PLOT=True
PLOT=False

SEQ_LEN=60


# In[101]:


url = "./datasets/DAT_ASCII_EURUSD_M1_2017.csv"
df = pd.read_csv(url, names=list(["date","open", "high", "low", "close", "volume"]), header=None, sep=";")
df.drop("volume", 1, inplace=True)
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
df.set_index('date', inplace=True)

df.head()


# In[102]:


if PLOT:
    df.plot(subplots=True, layout=(2, 2), figsize=(40, 20), sharex=False)


# In[103]:


data_set = df.iloc[:, 2:3].values #close
print(data_set)


# In[104]:


print(data_set.shape)


# In[105]:


sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(data_set)
print(data_set)


# In[106]:


X_train = []
y_train = []
for i in range(SEQ_LEN, training_set_scaled.size-1):
    X_train.append(training_set_scaled[i-SEQ_LEN:i, 0])
    y_train.append(training_set_scaled[i+1, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train[0])


# In[107]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train[0])


# In[108]:


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[ ]:




model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

opt = Adam(lr=0.001, decay=1e-6)

NAME = f"PRED-{int(time.time())}"  
# tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(optimizer=opt,loss='mean_squared_error')

model.fit(X_train,y_train,epochs=10,batch_size=32,validation_data=[X_test, y_test])

model.save('histdata.h5')

