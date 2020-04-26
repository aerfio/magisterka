from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import time

dataset_location = "/Users/i354746/private/magisterka/datasets/eurusd-m15-2018/EURUSD.csv"

df = pd.read_csv(dataset_location)

print(df.head())

plt.figure()
df.plot(x="Date", y="Close")
plt.show()
