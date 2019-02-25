#import json
import requests
#import urllib3
#import bs4 as bs
import numpy as np
import pandas as pd
from iex import Stock
#import matplotlib.pyplot as plt
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

#jeff bishop seminar notes:
# - 13 hourly ma and 30 hourly ma
# - 200 hourly ma is support (lock in when 13 and 30 goes down to 200)
# - take half at 100% then you cannot lose and you just let the other half run
# - don’t sell yourself short; let the chart run
# - look for overvaluation then 13/30 crossover and 200 ma safety
# - the farther 13/30 is above the 200 ma, the more room there is to correct
# - let yourself fail but don't lose more than 50% (stop 50% less than what you want to gain)
# - 2 or 4 week window for buying options (if you think move is going to happen in 2 weeks, buy for 4 weeks)
# - rsi over 80 means overbought
# - rsi under 20 means oversold

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
    file_name = './data/data-' + stock + '.csv'
    stock_data = init_get_data(stock, range_)
    rsi, up, down = rsi_func(stock, stock_data, 7, True)
    max_len = len(rsi)
    stock_data = stock_data.assign(rsi=rsi, up=up, down=down)

    ma_fast = ma_func(stock, stock_data, fast, True) #need to make max length of data (adding zeros before)
    ma_slow = ma_func(stock, stock_data, slow, True)
    ma_200 = ma_func(stock, stock_data, 200, True)
    ma_fast = np.concatenate([np.array([0]*(max_len-len(ma_fast))), ma_fast])
    ma_slow = np.concatenate([np.array([0]*(max_len-len(ma_slow))), ma_slow])
    ma_200 = np.concatenate([np.array([0] * (max_len - len(ma_200))), ma_200])
    stock_data = stock_data.assign(ma_fast=ma_fast, ma_slow=ma_slow, ma_200 = ma_200)

    ema_slow, ema_fast, macd, ema_macd = computeMACD(stock, stock_data, 30, 13, True)
    stock_data = stock_data.assign(ema_slow=ema_slow, ema_fast=ema_fast, macd=macd, ema_macd=ema_macd)
    #print(stock_data)

    stock_data.to_csv(file_name)

def retrieve_data(stock: str): #might need to take into consideration that the file does not exist
    file_name = './data/data-' + stock + '.csv'
    return pd.read_csv(file_name)

if __name__ == "__main__":

    #nyse_is_open situation (notes below might be irrelevant)
    # - if open, then run through
    # - if closed, then check if the previously added date to csv is equal to the prev date in iextrading

    # later want to check if csv exists, check if the file is up to date with the prev close as the last entry

    #current day data check will be below
    # - so if we put this into a while loop checking nyse_is_open, then if it becomes False, then go into "end-game mode"
    # - before the real-time can be run, the file for the stock must be checked

    #TO BE FIXED WITH ENTRY AND REQUEST SYSTEM
    stock_selected = 'CRON'#'DIS'
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
        line_chart.x_labels = stock_data['date'][-40:]
        line_chart.add('close', stock_data['close'][-40:])
        line_chart.add('ema_fast', stock_data['ema_fast'][-40:])
        line_chart.add('ema_slow', stock_data['ema_slow'][-40:])
        #line_chart.add('ma_fast', stock_data['ma_fast'][-40:])
        #line_chart.add('ma_slow', stock_data['ma_slow'][-40:])
        line_chart.add('ma_200', stock_data['ma_200'][-40:])
        line_chart.render_to_file('./assets/svg/chart_stock_price.svg')

        line_chart = pygal.Line()
        line_chart.title = 'RSI Threshold'
        line_chart.x_labels = stock_data['date'][-45:]
        line_chart.add('rsi', stock_data['rsi'][-45:])
        line_chart.add('upper_limit', np.array([70]*len(stock_data['date'][-45:])))
        line_chart.add('lower_limit', np.array([30]*len(stock_data['date'][-45:])))
        line_chart.render_to_file('./assets/svg/chart_rsi.svg')

        line_chart = pygal.Line()
        line_chart.title = 'MACD'
        line_chart.x_labels = stock_data['date'][-60:]
        line_chart.add('macd', stock_data['macd'][-60:])
        line_chart.add('ema_macd', stock_data['ema_macd'][-60:])
        line_chart.render_to_file('./assets/svg/chart_macd.svg')