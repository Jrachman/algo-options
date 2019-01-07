import json
import requests
import urllib3
import bs4 as bs
import numpy as np
import pandas as pd
from iex import Stock
import matplotlib.pyplot as plt

#what needs to be saved for specific day:
# - date
# - closing price
# - up
# - down
# - rsi
# - sma
# - ema
# - macd (through computeMACD function) -> might not need?

'''
for reference:
    
file_name = 'data-' + stock + '.csv'
data = pd.read_csv(file_name)
'''

def init_get_data(stock: str, range_: str):
    stock_chart = Stock(stock).chart_table(range=range_)
    return stock_chart[['date', 'close']] #maybe add change?

def init_rsi_func(prices, n=14): 
    deltas = np.diff(prices) #this can be replaced by just fetching change in init_get_data above
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100 - 100 / (1 + rs)
    #part above takes care of the initialization of the rsi function from 1 to n (14), then the rsi calculation begins
    for i in range(n, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n
        rs = up / down
        rsi[i] = 100 - 100 / (1 + rs)
        #note: in order to calculate today's rsi, you need to have 
        # (1) the difference between today's current/closing price to the previous day's closing price
        # (2) the previous up and down
    temp_up = np.zeros_like(prices)
    temp_up[-1] = up
    up = temp_up
    temp_down = np.zeros_like(prices)
    temp_down[-1] = down
    down = temp_down
    return rsi, np.diff(pd.concat([pd.Series([0]), prices])), up, down

def ma_func(values, window):
    weigths = np.repeat(1, window) / window
    smas = np.convolve(values, weigths, 'valid')
    return smas

def ema_func(values, window):
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

def computeMACD(x, slow=26, fast=12):
    emaslow = ema_func(x, slow)
    emafast = ema_func(x, fast)
    macd = emafast - emaslow
    sma_of_macd = ma_func(macd, 10)
    return emaslow, emafast, macd, sma_of_macd

def init_data(stock: str, range_: str) -> None: #maybe change range_ to window?
    stock_data = init_get_data(stock, range_)
    rsi, deltas, up, down = init_rsi_func(stock_data['close'], 8)
    stock_data = stock_data.assign(rsi=rsi, deltas=deltas, up=up, down=down) 
    file_name = 'data-' + stock + '.csv'
    stock_data.to_csv(file_name, index=False)

if __name__ == "__main__":
    
    stock_selected = 'NFLX'
    init_data(stock_selected, '5y')

    '''
    plt.subplot(3, 1, 1)
    for data in use_data_macd(stock_selected)[2:-2]:
        plt.plot(use_data_macd(stock_selected)[0][-14:], data[-14:])
    plt.xticks(rotation=90)

    plt.subplot(3, 1, 2)
    plt.plot(use_data_macd(stock_selected)[0][-30:], use_data_macd(stock_selected)[1][-30:])
    plt.plot(use_data_macd(stock_selected)[0][-30:], np.array([30]*len(use_data_macd(stock_selected)[0]))[-30:])
    plt.plot(use_data_macd(stock_selected)[0][-30:], np.array([70]*len(use_data_macd(stock_selected)[0]))[-30:])
    plt.xticks(rotation=90)
    
    plt.subplot(3, 1, 3)
    for data in use_data_macd(stock_selected)[-2:]:
        plt.plot(use_data_macd(stock_selected)[0][-30:], data[-30:])
    plt.xticks(rotation=90)

    plt.show()
    '''