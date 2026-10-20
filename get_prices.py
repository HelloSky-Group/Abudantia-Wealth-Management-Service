import yfinance as yf
import time


def _download(ticker, start_date, end_date, retries=5, wait=5):
    """
    Downloads adjusted close prices, retrying on Yahoo errors or empty responses
    """
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False,
                               progress=False)
            if not data.empty:
                return data['Adj Close']
            print(f"Empty response from Yahoo (attempt {attempt}/{retries})")
        except ValueError as e:
            print(f"Yahoo error (attempt {attempt}/{retries}): {e}")
        if attempt < retries:
            time.sleep(wait)
    raise RuntimeError(f"Could not download {ticker} after {retries} attempts")


def get_stock_data(ticker, start_date, end_date):
    """
    Gets historical stock data of a given stock ticker between a given start and end date
    :param ticker: stock ticker, e.g. 'AAPL'
    :type ticker: str or list of str
    :param start_date: start date, e.g. '2010-01-01'
    :type start_date: str of date "YYYY-MM-DD"
    :param end_date: end date, e.g. '2020-01-01'
    :type end_date: str of date "YYYY-MM-DD"
    :return: pandas dataframe of adjusted close prices, also saved to stock_data.csv
    """
    stock_data = _download(ticker, start_date, end_date)
    stock_data.to_csv('stock_data.csv')
    return stock_data


def get_sp500(start_date, end_date):
    """
    Gets historical prices of SPY, the S&P 500 ETF, between a given start and end date
    :param start_date: start date for S&P 500 prices, e.g. '2010-01-01'
    :type start_date: str of date "YYYY-MM-DD"
    :param end_date: end date for S&P 500 prices, e.g. '2020-01-01'
    :type end_date: str of date "YYYY-MM-DD"
    :return: pandas dataframe of adjusted close prices, also saved to sp500_data.csv
    """
    sp500_data = _download("SPY", start_date, end_date, wait=10)
    sp500_data.to_csv('sp500_data.csv')
    return sp500_data


if __name__ == '__main__':
    get_stock_data('AAPL', '2010-01-01', '2020-01-01')
    # get_sp500('2010-01-01', '2020-01-01')
