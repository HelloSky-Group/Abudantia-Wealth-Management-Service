import pandas_datareader.data as pdr
import yfinance as yf
import time
yf.pdr_override()


def get_stock_data(ticker, start_date, end_date):
    """
    Gets historical stock data of a given stock ticker between a given start and end date
    :param ticker: stock ticker, e.g. 'AAPL'
    :type ticker: str or list of str
    :param start_date: start date, e.g. '2010-01-01'
    :type start_date: str of date "YYYY-MM-DD"
    :param end_date: end date, e.g. '2020-01-01'
    :type end_date: str of date "YYYY-MM-DD"
    :return: pandas dataframe of stock data or stock_data.csv
    """
    i = 1
    try:
        all_data = pdr.get_data_yahoo(ticker, start_date, end_date)
    except ValueError:
        print("ValueError, trying again")
        i += 1
        if i < 5:
            time.sleep(5)
            get_stock_data(ticker, start_date, end_date)
        else:
            print("Tried 5 times, Yahoo error. Trying again after 2 minutes")
            time.sleep(120)
            get_stock_data(ticker, start_date, end_date)
    stock_data = all_data['Adj Close']
    stock_data.to_csv('stock_data.csv')


def get_sp500(start_date, end_date):
    """
    Gets historical stock data of all S&P 500 stocks between a given start and end date
    :param start_date: start date for S&P 500 prices, e.g. '2010-01-01'
    :type start_date: str of date "YYYY-MM-DD"
    :param end_date: end date for S&P 500 prices, e.g. '2020-01-01'
    :type end_date: str of date "YYYY-MM-DD"
    :return: sp500_data.csv
    """
    i = 1
    try:
        sp500_all_data = pdr.get_data_yahoo("SPY", start_date, end_date)
    except ValueError:
        print("ValueError, trying again")
        i += 1
        if i < 5:
            time.sleep(10)
            get_stock_data(start_date, end_date)
        else:
            print("Tried 5 times, Yahoo error. Trying again after 2 minutes")
            time.sleep(120)
            get_stock_data(start_date, end_date)
    sp500_data = sp500_all_data['Adj Close']
    sp500_data.to_csv('sp500_data.csv')


if __name__ == '__main__':
    get_stock_data('AAPL', '2010-01-01', '2020-01-01')
    # get_sp500('2010-01-01', '2020-01-01')
