#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Import the necessary libraries
import matplotlib.dates as dates
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# from matplotlib.finance import candlestick_ohlc
from mpl_finance import candlestick_ohlc

import numpy as np

# The section of the Plotly library needed
import plotly.graph_objects as go

from matplotlib.pyplot import figure
# get_ipython().run_line_magic('matplotlib', 'inline')

# Read in S&P 500 and Campbell Soup Company Data


def getDataset():
    url = "./datasets/DAT_ASCII_EURUSD_M1_2018.csv"
    df = pd.read_csv(url, names=list(
        ["date", "open", "high", "low", "close", "volume"]), header=None, sep=";")
    # df.drop(columns=['open', 'high', 'low'], inplace=True)
    # pdlen=len(df)

    # df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    # df.set_axis(df['date'], inplace=True)
    # df.drop(columns=["volume", "low", "open", "high"], inplace=True)
    # df["Close"]= df["close"] # compatibility
    # df["Volume"]= df["volume"] # compatibility
    # df.drop(columns=['close'], inplace=True)
    # df["ewma_close"]=df["close"].ewm(4).mean()
    # df = df[58::60]
    df = df[:3000]
    return df


# In[7]:


cpb = getDataset()

# Make a copy of the data frame before converting the date
# column to Matplotlib's format
cpb_mpl = cpb.copy()

# Convert Date column to Matplotlib's float format
cpb_mpl.date = dates.datestr2num(cpb_mpl.date)

print(cpb_mpl["date"][:3])

# Create a list of lists where each inner-list represents
# one day's trading history
cpb_subset = cpb_mpl[['date', 'open', 'high', 'low', 'close', 'volume']]
cpb_list = [list(str(x)) for x in cpb_subset.values]


# In[ ]:


fig = go.Figure(data=go.Ohlc(x=cpb['date'],
                             open=cpb['open'],
                             high=cpb['high'],
                             low=cpb['low'],
                             close=cpb['close']))

# Add title and annotations
fig.update_layout(title_text='CPB From November 21, 2018 to November 21, 2019',
                  title={
                      'y': 0.9,
                      'x': 0.5,
                      'xanchor': 'center',
                      'yanchor': 'top'},
                  xaxis_rangeslider_visible=True, xaxis_title="Time", yaxis_title="Growth Rate Percentage")


fig.show()


# %%
