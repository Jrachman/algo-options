import json
import requests
import urllib3
import bs4 as bs
import numpy as np
import pandas as pd
from iex import Stock

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

def init_get_data(stock: str):
    stock_chart = Stock(stock).chart_table(range="1y")
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
    weigths = np.repeat(1.0, window) / window
    smas = np.convolve(values, weigths, 'valid')
    return smas # as a numpy array

def ema_func(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

def computeMACD(x, slow=26, fast=12):
    #compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    #return value is emaslow, emafast, macd which are len(x) arrays
    emaslow = ema_func(x, slow)
    emafast = ema_func(x, fast)
    return emaslow, emafast, emafast - emaslow

def init_data(stock: str) -> None:
    stock_data = init_get_data(stock)
    rsi, deltas, up, down = init_rsi_func(stock_data['close'], 8) #8 periods, instead of default 14; 70/30 is indicator for oversold, overbought
    stock_data = stock_data.assign(rsi=rsi, deltas=deltas, up=up, down=down) 
    file_name = 'data-' + stock + '.csv'
    stock_data.to_csv(file_name, index=False)

if __name__ == "__main__":
    my_stocks = ['SPY', 'AMZN', 'AMD', 'AAPL', 'NVDA', 'TSLA']
    #if len(my_stocks) == 0:
        #my_stocks = sp500_tickers()
    #for stock in my_stocks:
        #init_data(stock)
    #for _ in range(20):
        #print(Stock("F").price())
    print(nyse_is_open())