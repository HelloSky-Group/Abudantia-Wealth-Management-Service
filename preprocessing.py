import pandas as pd
import numpy as np


class DataProcessing:
    def __init__(self, file, train, column=None):
        """
        Splits a price series into chronological train and test sets
        :param file: path to a csv with a date column first, e.g. stock_data.csv
        :param train: fraction of rows used for training, e.g. 0.8
        :param column: price column to use; defaults to the first column after the date
        """
        self.y_test = None
        self.x_test = None
        self.y_train = None
        self.x_train = None
        self.file = pd.read_csv(file)
        self.column = column if column is not None else self.file.columns[1]
        self.train = train
        self.i = int(self.train * len(self.file))
        self.stock_train = self.file[0:self.i]
        self.stock_test = self.file[self.i:]
        self.min = None
        self.max = None

    @staticmethod
    def _windows(series, seq_len):
        """
        Builds windows of seq_len prices, each paired with the price that follows it
        """
        x = np.array([series[i:i + seq_len] for i in range(len(series) - seq_len)], np.float64)
        y = np.array([[series[i + seq_len]] for i in range(len(series) - seq_len)], np.float64)
        return x, y

    def fit_scaler(self):
        """
        Fits min-max scaling on the training set only, so no test data leaks into training
        """
        prices = self.stock_train[self.column].to_numpy(np.float64)
        self.min = prices.min()
        self.max = prices.max()

    def scale(self, values):
        return (values - self.min) / (self.max - self.min)

    def unscale(self, values):
        return values * (self.max - self.min) + self.min

    def gen_train(self, seq_len):
        """
        Generates training data
        :param seq_len: length of window
        :return: x_train and y_train
        """
        series = self.stock_train[self.column].to_numpy(np.float64)
        self.x_train, self.y_train = self._windows(series, seq_len)

    def gen_test(self, seq_len):
        """
        Generates testing data
        :param seq_len: length of window
        :return: x_test and y_test
        """
        series = self.stock_test[self.column].to_numpy(np.float64)
        self.x_test, self.y_test = self._windows(series, seq_len)
