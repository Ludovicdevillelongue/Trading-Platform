from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_loader.data_retriever import DataRetriever
from sklearn import linear_model
import warnings
# To deactivate all warnings:
warnings.filterwarnings('ignore')

# A base backtesting class with common functionality
class StrategyCreator:
    def __init__(self, symbol, start_date, end_date, amount, transaction_costs):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.amount = amount
        self.transaction_costs = transaction_costs
        self.data = None  # Will hold historical price data
        self.results = None  # Will hold backtest results

    def get_data(self):
        ''' Retrieves and prepares the data'''
        # raw = pd.read_hdf(DataRetriever(self.start_date, self.end_date).read_data())
        raw=DataRetriever(self.start_date, self.end_date).yfinance_download(self.symbol)['price'].to_frame()
        # DataRetriever(self.start_date, self.end_date).write_data(raw)
        # raw.rename(columns={self.symbol.split('/')[1]: 'price'}, inplace=True)
        raw['return'] = np.log(raw / raw.shift(1))
        return raw

    def select_data(self, start, end):
        ''' Selects sub-sets of the financial data.
        '''
        data = self.data[(self.data.index >= start) &
                         (self.data.index <= end)].copy()
        return data

    def set_train_test_split(self, train_percent):
        ''' Sets the training and testing periods based on a percentage split '''
        if train_percent <= 0 or train_percent >= 1:
            raise ValueError("train_percent must be between 0 and 1.")

        total_data_points = len(self.data)
        split_point = int(total_data_points * train_percent)

        self.train_start = self.data.index[0]
        self.train_end = self.data.index[split_point]
        self.test_start = self.data.index[split_point + 1]
        self.test_end = self.data.index[-1]


    def prepare_lags(self, start, end, lags):
        data = self.select_data(start, end)
        self.cols = []
        for lag in range(1, lags + 1):
            col = f'lag_{lag}'
            data[col] = data['return'].shift(lag)
            self.cols.append(col)
        data.dropna(inplace=True)  # Drop rows with NaN values
        self.lagged_data = data



    def fit_model(self, start, end, lags, model):
        ''' Implements the fitting step.
        '''

        self.prepare_lags(start, end, lags)
        if model == 'linalg':
            self.reg = np.linalg.lstsq(self.lagged_data[self.cols],
                                       np.sign(self.lagged_data['return']),
                                       rcond=None)[0]
        else:
            model.fit(self.lagged_data[self.cols],
                      np.sign(self.lagged_data['return']))

    def prepare_lags_for_signal(self, lags):
        ''' Prepares the lagged data for the most recent data point for signal generation. '''
        latest_data = self.data.tail(2*lags + 1).copy()  # Get enough rows for lags
        cols = [f'lag_{lag}' for lag in range(1, lags + 1)]
        for lag in range(1, lags + 1):
            latest_data[f'lag_{lag}'] = latest_data['return'].shift(lag)
        latest_data.dropna(inplace=True)
        return latest_data, cols

    def fit_model_for_signal(self, lags, model):
        latest_data, cols = self.prepare_lags_for_signal(lags)
        latest_row = latest_data.tail(1)
        if model == 'linalg':
            self.reg = np.linalg.lstsq(latest_row[cols],
                                       np.sign(latest_row['return']),
                                       rcond=None)[0]
            return latest_row
        else:
            model.fit(latest_row[cols], np.sign(latest_row['return']))
            return latest_row


    def calculate_performance(self, data):
        """ Calculate performance and return a DataFrame with results. """

        # Calculate strategy return
        data['strategy'] = data['position'].shift(1) * data['return']
        data = data.dropna(axis=0)

        # Subtract transaction costs from return when trade takes place
        data.loc[data['orders']!=0, 'strategy'] -= self.transaction_costs*abs(data['orders'])

        # Annualize the mean log return
        data['an_mean_log_returns'] = data[['return', 'strategy']].mean() * 252
        # Convert log returns to regular returns for comparison
        data['an_mean_returns'] = np.exp(data['an_mean_log_returns']) - 1

        # Annualize the standard deviation of log returns
        data['an_std_log_returns'] = data[['return', 'strategy']].std() * 252 ** 0.5

        # Calculate cumulative returns
        data['creturns'] = self.amount * data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = self.amount * data['strategy'].cumsum().apply(np.exp)

        # Save results for further output or plot
        self.results = data

        # Calculate absolute and out-/underperformance
        aperf = self.results['cstrategy'].iloc[-1]
        operf = aperf - self.results['creturns'].iloc[-1]

        # Calculate the Sharpe Ratio
        try:
            risk_free_rate = 0.01  # Risk-free rate of return
            data['excess_return'] = data['strategy'] - risk_free_rate / 252
            sharpe_ratio = (data['excess_return'].mean() / data['excess_return'].std()) * np.sqrt(252)
        except Exception as e:
            sharpe_ratio=0
        return round(aperf, 2), round(operf, 2), round(sharpe_ratio, 2)

    def analyse_strategy(self, data):
        # Placeholder for strategy analysis method
        raise NotImplementedError("Should implement method in subclass!")

    def run_strategy(self):
        # Placeholder for strategy execution method
        raise NotImplementedError("Should implement method in subclass!")

    def generate_signal(self):
        raise NotImplementedError("Should implement method in subclass!")

# Now define specific strategy backtesters
class SMAVectorBacktester(StrategyCreator):
    def __init__(self, data, symbol, start_date, end_date, sma_short, sma_long, amount, transaction_costs):
        super().__init__(symbol, start_date, end_date, amount, transaction_costs)
        self.data=data
        self.sma_short = sma_short
        self.sma_long = sma_long

    def set_parameters(self, SMA1=None, SMA2=None):
        ''' Updates SMA parameters and resp. time series.
         '''
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['price'].rolling(self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()

    def analyse_strategy(self, data):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["price"].plot(ax=ax1, color='g', lw=.5)
        data["SMA1"].plot(ax=ax1, color='r', lw=2.)
        data["SMA2"].plot(ax=ax1, color='b', lw=2.)
        ax1.plot(data.loc[data.orders == 2.0].index,
                 data["price"][data.orders == 2.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders == -2.0].index,
                 data["price"][data.orders == -2.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Short mavg", "Long mavg", "Buy", "Sell"])
        plt.title("Simple Moving Average Trading Strategy")
        # plt.show()



    def run_strategy(self):

        '''
        Backtests the trading strategy.
        '''
        self.set_parameters(int(self.sma_short), int(self.sma_long))
        data = self.data.copy().dropna()
        data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        self.data_signal=data['position'][-1]
        self.analyse_strategy(data)
        return self.calculate_performance(data)

    def generate_signal(self):
        self.run_strategy()
        return self.data_signal


class MomVectorBacktester(StrategyCreator):
    def __init__(self, data, symbol, start_date, end_date, momentum, amount, transaction_costs):
        super().__init__(symbol, start_date, end_date, amount, transaction_costs)
        self.data=data
        self.momentum = momentum

    def analyse_strategy(self, data):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["price"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders == 2.0].index,
                 data["price"][data.orders == 2],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders == -2.0].index,
                 data["price"][data.orders == -2],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Buy", "Sell"])
        plt.title("Momentum Trading Strategy")
        # plt.show()

    def run_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data['position'] = np.sign(data['return'].rolling(self.momentum).mean())
        self.data_signal=data['position'][-1]
        self.analyse_strategy(data)
        return self.calculate_performance(data)

    def generate_signal(self):
        self.run_strategy()
        return self.data_signal

class MRVectorBacktester(StrategyCreator):
    def __init__(self, data, symbol, start_date, end_date, sma, threshold, amount, transaction_costs):
        super().__init__(symbol, start_date, end_date, amount, transaction_costs)
        self.data=data
        self.sma=sma
        self.threshold=threshold

    def set_parameters(self, SMA=None):
        if SMA is not None:
            self.data['sma'] = self.data['price'].rolling(SMA).mean()

    def analyse_strategy(self, data):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["price"].plot(ax=ax1, color='g', lw=.5)
        data["sma"].plot(ax=ax1, color='b', lw=2.)
        ax1.plot(data.loc[data.orders == 1.0].index,
                 data["price"][data.orders == 1],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders == -1.0].index,
                 data["price"][data.orders == -1],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "SMA", "Buy", "Sell"])
        plt.title("Mean Reversion Trading Strategy")
        # plt.show()

    def run_strategy(self):
        self.set_parameters(int(self.sma))
        data = self.data.copy().dropna()
        data['distance'] = data['price'] - data['sma']
        # sell signals
        data['position'] = np.where(data['distance'] > self.threshold,
                                    -1, np.nan)
        # buy signals
        data['position'] = np.where(data['distance'] < -self.threshold,
                                    1, data['position'])
        # crossing of current price and SMA (zero distance)
        data['position'] = np.where(data['distance'] *
                                    data['distance'].shift(1) < 0,
                                    0, data['position'])
        data['position'] = data['position'].ffill().fillna(0)
        self.data_signal=data['position'][-1]
        self.analyse_strategy(data)
        return self.calculate_performance(data)

    def generate_signal(self):
        self.run_strategy()
        return self.data_signal


class TurtleVectorBacktester(StrategyCreator):
    def __init__(self, data, symbol, start_date, end_date, window_size, amount, transaction_costs):
        super().__init__(symbol, start_date, end_date, amount, transaction_costs)
        self.data=data
        self.window_size=window_size


    def analyse_strategy(self, data):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data["price"].plot(ax=ax1, color='g', lw=.5)
        data["high"].plot(ax=ax1, color='g', lw=2.)
        data["low"].plot(ax=ax1, color='r', lw=2.)
        data["avg"].plot(ax=ax1, color='b', lw=2.)
        ax1.plot(data.loc[data.orders == 1.0].index,
                 data["price"][data.orders == 1],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders == -1.0].index,
                 data["price"][data.orders == -1],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Highs", "Lows", "Average", "Buy", "Sell"])
        plt.title("Turtle Trading Strategy")
        # plt.show()

    def run_strategy(self):
        data = self.data.copy().dropna()
        # window_size-days high
        data['high'] = data['price'].shift(1). \
        rolling(window=self.window_size).max()
        # window_size-days low
        data['low'] = data['price'].shift(1). \
        rolling(window=self.window_size).min()
        # window_size-days mean
        data['avg'] = data['price'].shift(1). \
        rolling(window=self.window_size).mean()
        data['long_entry'] = data['price'] > data.high
        data['short_entry'] = data['price'] < data.low
        data['long_exit'] = data['price'] < data.avg
        data['short_exit'] = data['price'] > data.avg
        data["position"] = 0
        data["orders"]=0
        for k in range(1, len(data)):
            if data['long_entry'][k] and data['position'][k-1] == 0:
                data.orders.values[k] = 1
                data.position.values[k] = 1
            elif data['short_entry'][k] and data['position'][k-1] == 0:
                data.orders.values[k] = -1
                data.position.values[k] = -1
            elif data['short_exit'][k] and data['position'][k-1] > 0:
                data.orders.values[k] = -1
                data.position.values[k] = 0
            elif data['long_exit'][k] and data['position'][k-1] < 0:
                data.orders.values[k] = 1
                data.position.values[k] = 0
            else:
                data.orders.values[k] = 0
        self.data_signal=data['position'][-1]
        # self.analyse_strategy(data)
        return self.calculate_performance(data)

    def generate_signal(self):
        self.run_strategy()
        return self.data_signal

class LRVectorBacktester(StrategyCreator):

    def __init__(self, data, symbol, start_date, end_date, lags, train_percent, amount, transaction_costs):
        super().__init__(symbol, start_date, end_date, amount, transaction_costs)
        self.data=data
        self.lags=lags
        self.cols = [f'lag_{lag}' for lag in range(1, self.lags + 1)]
        self.train_percent=train_percent
        self.model='linalg'
        self.train_start = None
        self.train_end = None
        self.test_start = None
        self.test_end = None

    def analyse_strategy(self, data):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["price"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders == 2.0].index,
                 data["price"][data.orders == 2],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders == -2.0].index,
                 data["price"][data.orders == -2],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Buy", "Sell"])
        plt.title("Linear Regression Prediction Trading Strategy")
        # plt.show()


    def run_strategy(self):
        ''' Backtests the trading strategy. '''
        self.set_train_test_split(self.train_percent)
        self.fit_model(self.train_start, self.train_end, self.lags, self.model)

        # Using the entire test period
        self.results = self.select_data(self.test_start, self.test_end)

        # Prepare lags for the testing set
        self.prepare_lags(self.test_start, self.test_end, self.lags)
        prediction = np.sign(np.dot(self.lagged_data[self.cols], self.reg))

        # Initialize predictions with 0 for the period without enough lags
        self.results['position'] = 0
        # Fill in predictions where we have enough lagged data
        self.results.loc[self.lagged_data.index, 'position'] = prediction
        self.analyse_strategy(self.results)
        # Call calculate_performance from parent class
        aperf, operf, sharpe_ratio = self.calculate_performance(self.results)

        return round(aperf, 2), round(operf, 2), round(sharpe_ratio, 2)


    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        latest_row=self.fit_model_for_signal(self.lags, self.model)
        if not latest_row.empty and self.reg is not None:
            prediction = np.dot(latest_row[self.cols], self.reg)
            return int(np.sign(prediction))
        else:
            return 0  # Return neutral signal if data is insufficient


class ScikitVectorBacktester(StrategyCreator):
    def __init__(self, data, symbol, start_date, end_date, lags, train_percent, model, amount, transaction_costs):
        super().__init__(symbol, start_date, end_date, amount, transaction_costs)
        self.data=data
        self.lags=lags
        self.cols = [f'lag_{lag}' for lag in range(1, self.lags + 1)]
        self.train_percent=train_percent
        self.model=model
        self.train_start = None
        self.train_end = None
        self.test_start = None
        self.test_end = None

        if model == 'regression':
            self.model = linear_model.LinearRegression()
        elif model == 'logistic':
            self.model = linear_model.LogisticRegression(C=1e6,
                                                         solver='lbfgs',
                                                         multi_class='ovr',
                                                         max_iter=1000)
        else:
            raise ValueError('Model not known or not yet implemented.')


    def run_strategy(self):
        ''' Backtests the trading strategy. '''
        self.set_train_test_split(self.train_percent)
        self.fit_model(self.train_start, self.train_end, self.lags, self.model)

        # Using the entire test period
        self.results = self.select_data(self.test_start, self.test_end)

        # Prepare lags for the testing set
        self.prepare_lags(self.test_start, self.test_end, self.lags)
        prediction = self.model.predict(self.lagged_data[self.cols])

        # Initialize predictions with 0 for the period without enough lags
        self.results['position'] = 0
        # Fill in predictions where we have enough lagged data
        self.results.loc[self.lagged_data.index, 'position'] = prediction
        # Call calculate_performance from parent class
        aperf, operf, sharpe_ratio = self.calculate_performance(self.results)

        return round(aperf, 2), round(operf, 2), round(sharpe_ratio, 2)


    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        latest_row=self.fit_model_for_signal(self.lags, self.model)
        if not latest_row.empty:
            prediction = self.model.predict(latest_row[self.cols])
            return int(np.sign(prediction))
        else:
            return 0



