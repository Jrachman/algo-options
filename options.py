import json
import requests
import urllib3
import bs4 as bs
import numpy as np
import pandas as pd
from iex import Stock
import matplotlib.pyplot as plt

#personal notes:
# - periods are based on what the increments on the charted data are (i.e., if 1m is based on day, then the period will be days)

#jeff bishop seminar notes:
# - 13 hourly ma and 30 hourly ma
# - 200 hourly ma is support (lock in when 13 and 30 goes down to 200)
# - take half at 100% then you cannot lose and you just let the other half run
# - don’t sell yourself short; let the chart run
# - look for overvaluation then 13/30 crossover and 200 ma safety
# - the farther 13/30 is above the 200 ma, the more room there is to correct
# - let yourself fail but don't lose more than 50% (stop 50% less than what you want to gain)
# - 2 or 4 week window foxr buying options (if you think move is going to happen in 2 weeks, buy for 4 weeks)
# - rsi over 80 means overbought
# - rsi under 20 means oversold

#need to do:
# - create funct for the current trading day and checking the current close and calculating
# - create global for opening csv files given stock

def sp500_tickers() -> [str]:
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    return tickers

def nyse_is_open() -> str: #return whether or not the stock market is open right now. if it is, then continue running application; else, stop feeding data as long as market is closed
    response = requests.get("https://www.stockmarketclock.com/api-v1/status?exchange=nyse")
    return response.json()['results']['nyse']['status']

def test_for_hourly_analysis(): #note that this function should be based on the range of init_get_data
    response = requests.get("https://api.iextrading.com/1.0/stock/aapl/chart/1d")
    list_of_dicts = response.json()
    time_and_close = {'minute': [], 'close': []}
    for dict_ in list_of_dicts:
        if int(dict_['minute'][-2:]) == 59:
            time_and_close['minute'].append(dict_['minute'])
            time_and_close['close'].append(dict_['close']) 
    return time_and_close


def init_get_data(stock: str, range_: str):
    stock_chart = Stock(stock).chart_table(range=range_)
    return stock_chart[['date', 'close']]

def init_rsi_func(prices, n=14):
    deltas = np.diff(prices) #out[n] = a[n+1] - a[n]
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
    return smas # as a numpy array

def ema_func(values, window):
    weights = np.exp(np.linspace(-1, 0, window))
    #print(weights)
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

def computeMACD(x, slow=26, fast=12):
    #compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    #return value is emaslow, emafast, macd which are len(x) arrays
    emaslow = ema_func(x, slow)
    emafast = ema_func(x, fast)
    macd = emafast - emaslow
    sma_of_macd = ma_func(macd, 10)
    return emaslow, emafast, macd, sma_of_macd

def init_data(stock: str, range_: str) -> None:
    stock_data = init_get_data(stock, range_)
    rsi, deltas, up, down = init_rsi_func(stock_data['close'], 8) #8 periods, instead of default 14; 70/30 is indicator for oversold, overbought
    stock_data = stock_data.assign(rsi=rsi, deltas=deltas, up=up, down=down) 
    file_name = 'data-' + stock + '.csv'
    stock_data.to_csv(file_name, index=False)

def use_data(stock): #still a testing func
    file_name = 'data-' + stock + '.csv'
    data = pd.read_csv(file_name)
    list_of_periods = [1, 13, 30, 200]
    max_len = len(data['close'])
    smas = [data['date'], data['rsi']]
    for i in list_of_periods:
        sma = ma_func(data['close'], i)
        temp = np.concatenate([np.array([0]*(max_len-len(sma))), sma])
        smas.append(temp)
    return smas

def current_day_calc(stock, n=14): #STILL IN THE WORKS!; would add nyse_is_open but will do that later
    current_price = Stock(stock).price()
    file_name = 'data-' + stock + '.csv'
    data = pd.read_csv(file_name)

    #rsi
    delta = current_price - data['close'].iloc[-1]
    up = data['up'].iloc[-1]
    down = data['down'].iloc[-1]
    if delta > 0:
        upval = delta
        downval = 0
    else:
        upval = 0
        downval = -delta
    up = (up * (n - 1) + upval) / n
    down = (down * (n - 1) + downval) / n
    rs = up / down
    rsi = 100 - 100 / (1 + rs)
    #print(current_price, up, down, rs, rsi) #save up, down, and rsi
    
    test_data = pd.concat([data['close'], pd.Series([current_price])])
    slow, fast, macd, sma_of_macd = computeMACD(test_data, slow=30, fast=13)
    max_len = len(data['close'])
    emas = [data['date'], data['rsi'], data['close']]
    for i in [slow, fast, macd, sma_of_macd]:
        temp = np.concatenate([np.array([0]*(max_len-len(i))), i])
        emas.append(temp)
    #print(emas)
    return current_price, up, down, rsi, emas

def use_data_macd(stock):
    file_name = 'data-' + stock + '.csv'
    data = pd.read_csv(file_name)
    slow, fast, macd, sma_of_macd = computeMACD(data['close'], slow=30, fast=13)
    max_len = len(data['close'])
    emas = [data['date'], data['rsi'], data['close']]
    for i in [slow, fast, macd, sma_of_macd]:
        temp = np.concatenate([np.array([0]*(max_len-len(i))), i])
        emas.append(temp)
    return emas

if __name__ == "__main__":
    my_stocks = ['CRON']#, 'SPY', 'AMZN', 'AMD', 'AAPL', 'NVDA', 'TSLA']
    #if len(my_stocks) == 0:
        #my_stocks = sp500_tickers()
        
    for stock in my_stocks:
        init_data(stock, '5y') #change range for iextrading api here

    #print(nyse_is_open())
    #print(test_for_hourly_analysis())

    #'''
    stock_selected = 'CRON'
    print(current_day_calc(stock_selected)) #NEW ALGO FOR CURRENT REALTIME!

    #'''
    plt.subplot(3, 1, 1) #replace with new computeMACD funct (done)
    for data in use_data_macd(stock_selected)[2:-2]:
        plt.plot(use_data_macd(stock_selected)[0][-14:], data[-14:])
    #for data in use_data(stock_selected)[3:]:
        #plt.plot(use_data(stock_selected)[0], data)
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
    #'''
    #paste new realtime graph here
    '''

    plt.subplot(2, 1, 1) #replace with new computeMACD funct (done)
    for data in use_data('AAPL')[2:]:
        plt.plot(use_data('AAPL')[0], data)
    plt.xticks(rotation=90)

    plt.subplot(2, 1, 2)
    plt.plot(use_data_macd('AAPL')[0], use_data_macd('AAPL')[1])
    plt.plot(use_data('AAPL')[0], np.array([30]*len(use_data('AAPL')[0])))
    plt.plot(use_data('AAPL')[0], np.array([70]*len(use_data('AAPL')[0])))
    plt.xticks(rotation=90)

    plt.show()
    '''
    #for _ in range(20):
        #print(Stock("F").price())