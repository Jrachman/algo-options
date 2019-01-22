#import json
import requests
#import urllib3
#import bs4 as bs
import numpy as np
import pandas as pd
from iex import Stock
import matplotlib.pyplot as plt
from datetime import date
import pygal

#what graphs are needed:
#  graph #1:
#    - stock price
#    - slow ema
#    - fast ema
#  graph #2:
#    - rsi
#    - upper-limit line (70 or 80)
#    - lower-limit line (30 or 20)
#  graph #3:
#    - macd
#    - ma of macd

#what needs to be saved for specific day:
# - date
# - closing price
# - up
# - down
# - rsi
# - sma
# - ema
# - macd (through computeMACD function) -> might not need?

#important links for ma and ema:
# - https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
# - https://en.wikipedia.org/wiki/Moving_average

def nyse_is_open() -> str: #return whether or not the stock market is open right now. if it is, then continue running application; else, stop feeding data as long as market is closed
    response = requests.get("https://www.stockmarketclock.com/api-v1/status?exchange=nyse")
    return response.json()['results']['nyse']['status']

def init_get_data(stock: str, range_: str):
    stock_chart = Stock(stock).chart_table(range=range_)
    return stock_chart[['date', 'close', 'change']]

def rsi_func(stock, data, n=14, init=False): 
    prices = data['close']
    if init == True:
        deltas = data['change'] #this can be replaced by just fetching change in init_get_data above
        seed = deltas[:n + 1]
        up = seed[seed >= 0].sum() / n
        down = -seed[seed < 0].sum() / n
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:n] = 100 - 100 / (1 + rs)
        #part above takes care of the initialization of the rsi function from 1 to n (14), then the rsi calculation begins
        for i in range(n, len(prices)):
            delta = deltas[i - 1]
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
        return rsi, up, down
    elif init == False:
        current_price = Stock(stock).price() #the only use for the stock param
        delta = current_price - prices.iloc[-1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
        up = (data['up'].iloc[-1] * (n - 1) + upval) / n
        down = (data['down'].iloc[-1] * (n - 1) + downval) / n
        rs = up / down
        rsi = 100 - 100 / (1 + rs)
        return current_price, rsi, up, down
    #else:
        #return 'string return response here'


def ma_func(stock, data, window, init=False): #next_ma = prev_ma + (current_price / window) - (price_window_days_ago / window)
    if init == True:
        weigths = np.repeat(1, window) / window
        smas = np.convolve(data['close'], weigths, 'valid')
        return smas
    elif init == False:
        current_price = Stock(stock).price()
        prev_ma = data['ma_macd'].iloc[-1] #here is the last moving average for the csv
        top_window_ma = data['ma_macd'].iloc[-window]
        curr_ma = prev_ma + (current_price / window) - (top_window_ma / window)
        return curr_ma

def ema_func(stock, data, window, speed, init=False):
    if init == True:
        weights = np.exp(np.linspace(-1, 0, window))
        weights /= weights.sum()
        a =  np.convolve(data['close'], weights, mode='full')[:len(data['close'])]
        a[:window] = a[window]
        return a
    elif init == False:
        if speed == 'fast':
            prev_ema = data['ema_fast'].iloc[-1]
        elif speed == 'slow':
            prev_ema = data['ema_slow'].iloc[-1]
        elif speed == 'ema_macd':
            prev_ema = data['ema_macd'].iloc[-1]
        current_price = Stock(stock).price() #note that if it is neither of the speeds, then this part and beyond will fail
        weight = 2 / (window + 1)
        curr_ema = (current_price - prev_ema) * weight + prev_ema
        return curr_ema
        

def computeMACD(stock, x, slow=26, fast=12, init=False):
    emaslow = ema_func(stock, x, slow,'slow', init)
    emafast = ema_func(stock, x, fast, 'fast', init)
    macd = emafast - emaslow
    if init == True:
        weigths = np.repeat(1, 9) / 9
        ema_of_macd = np.convolve(macd, weigths, 'valid')
        ema_of_macd = np.concatenate([np.array([0]*(len(emaslow)-len(ema_of_macd))), ema_of_macd])
        return emaslow, emafast, macd, ema_of_macd
    elif init == False:
        ema_of_macd = ema_func(stock, x, 9, 'ema_macd')
        return emaslow, emafast, macd, ema_of_macd


def init_data(stock: str, range_: str, fast: int, slow: int) -> None: #maybe change range_ to window?
    file_name = 'data-' + stock + '.csv'
    stock_data = init_get_data(stock, range_)
    rsi, up, down = rsi_func(stock, stock_data, 8, True)
    max_len = len(rsi)
    stock_data = stock_data.assign(rsi=rsi, up=up, down=down)

    ma_fast = ma_func(stock, stock_data, fast, True) #need to make max length of data (adding zeros before)
    ma_slow = ma_func(stock, stock_data, slow, True)
    ma_fast = np.concatenate([np.array([0]*(max_len-len(ma_fast))), ma_fast])
    ma_slow = np.concatenate([np.array([0]*(max_len-len(ma_slow))), ma_slow])
    stock_data = stock_data.assign(ma_fast=ma_fast, ma_slow=ma_slow)

    ema_slow, ema_fast, macd, ema_macd = computeMACD(stock, stock_data, 30, 13, True)
    stock_data = stock_data.assign(ema_slow=ema_slow, ema_fast=ema_fast, macd=macd, ema_macd=ema_macd)
    #print(stock_data)

    stock_data.to_csv(file_name)

def retrieve_data(stock: str): #might need to take into consideration that the file does not exist
    file_name = 'data-' + stock + '.csv'
    return pd.read_csv(file_name)

if __name__ == "__main__":
    #nyse_is_open situation (notes below might be irrelevant)
    # - if open, then run through
    # - if closed, then check if the previously added date to csv is equal to the prev date in iextrading

    # later want to check if csv exists, check if the file is up to date with the prev close as the last entry

    #current day data check will be below
    # - so if we put this into a while loop checking nyse_is_open, then if it becomes False, then go into "end-game mode"
    # - before the real-time can be run, the file for the stock must be checked

    stock_selected = 'CRON'
    init_data(stock_selected, '5y', 13, 30)

    curr_data = retrieve_data(stock_selected)
    current_price, rsi, up, down = rsi_func(stock_selected, curr_data, 14)
    emaslow, emafast, macd, ema_of_macd = computeMACD(stock_selected, curr_data, 30, 13)

    #pygal graphing
    stock_data = retrieve_data(stock_selected)

    if nyse_is_open() != False:
        today = str(date.today())

        line_chart = pygal.Line()
        line_chart.title = 'Stock Price with EMA'
        line_chart.x_labels = stock_data['date'][-60:]
        line_chart.add('close', stock_data['close'][-60:])
        line_chart.add('ema_slow', stock_data['ema_slow'][-60:])
        line_chart.add('ema_fast', stock_data['ema_fast'][-60:])
        line_chart.render_in_browser()

        line_chart = pygal.Line()
        line_chart.title = 'RSI Threshold'
        line_chart.x_labels = stock_data['date'][-60:]
        line_chart.add('rsi', stock_data['rsi'][-60:])
        line_chart.add('upper_limit', np.array([70]*len(stock_data['date'][-60:])))
        line_chart.add('lower_limit', np.array([30]*len(stock_data['date'][-60:])))
        line_chart.render_in_browser()

        line_chart = pygal.Line()
        line_chart.title = 'MACD'
        line_chart.x_labels = stock_data['date'][-60:]
        line_chart.add('macd', stock_data['macd'][-60:])
        line_chart.add('ema_macd', stock_data['ema_macd'][-60:])
        line_chart.render_in_browser()

    #matplotlib graphing
    '''
    if nyse_is_open() == True:
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