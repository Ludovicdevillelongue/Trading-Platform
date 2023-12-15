import matplotlib.pyplot as plt
import numpy as np
from data_loader.data_retriever import DataRetriever
from sklearn import linear_model
import warnings
# To deactivate all warnings:
warnings.filterwarnings('ignore')

# A base backtesting class with common functionality
class StrategyCreator:
    def __init__(self, frequency, symbol, start_date, end_date, amount, transaction_costs):
        self.frequency=frequency
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.amount = amount
        self.transaction_costs = transaction_costs
        self.data = None  # Will hold historical price data
        self.results = None  # Will hold backtest results

    def get_data(self, data_provider):
        ''' Retrieves and prepares the data'''
        if data_provider=='yfinance':
            # raw = pd.read_hdf(DataRetriever(self.start_date, self.end_date).read_data())
            raw=DataRetriever(self.frequency, self.start_date, self.end_date).yfinance_download(self.symbol)\
            [['open', 'high', 'low',  'close', 'volume']]
            # DataRetriever(self.start_date, self.end_date).write_data(raw)
            # raw.rename(columns={self.symbol.split('/')[1]: 'price'}, inplace=True)
            raw['return'] = np.log(raw['close'] / raw['close'].shift(1))
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
        risk_free_rate = 0.01
        try:
             # Risk-free rate of return
            data['excess_return'] = data['strategy'] - risk_free_rate / 252
            sharpe_ratio = (data['excess_return'].mean() / data['excess_return'].std()) * np.sqrt(252)
        except Exception as e:
            sharpe_ratio=0

        # Sortino Ratio
        negative_std = np.std(data.loc[data['strategy'] < 0, 'strategy']) * np.sqrt(252)
        sortino_ratio = (data['strategy'].mean() - risk_free_rate / 252) / negative_std if negative_std != 0 else 0

        # Drawdown
        roll_max = data['cstrategy'].cummax()
        drawdown = data['cstrategy'] / roll_max - 1.0
        data['drawdown'] = drawdown

        # Maximum Drawdown
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar_ratio = data['strategy'].mean() * 252 / abs(max_drawdown) if max_drawdown != 0 else 0

        # Alpha and Beta (compared to 'data['return']' as the benchmark)
        covariance = np.cov(data['strategy'], data['return'])[0][1]
        beta = covariance / np.var(data['return'])
        alpha = (data['strategy'].mean() - risk_free_rate / 252) - beta * (data['return'].mean() - risk_free_rate / 252)

        return round(aperf, 0), round(operf, 0), round(sharpe_ratio, 2),round(sortino_ratio, 2), round(calmar_ratio, 2),\
               round(max_drawdown, 0), round(alpha, 2), round(beta, 2)


    def position_holding_time(self, data):
        position_holding_times = []
        current_pos = 0
        current_pos_start = 0
        for i in range(0, len(data)):
            pos = data['position'].iloc[i]
            # flat and starting a new position
            if current_pos == 0:
                if pos != 0:
                    current_pos = pos
                    current_pos_start = i
                continue
            # going from long position to flat or short position or
            # going from short position to flat or long position
            if current_pos * pos <= 0:
                current_pos = pos
                position_holding_times.append(i - current_pos_start)
                current_pos_start = i
        print(position_holding_times)
        plt.hist(position_holding_times, 100)
        plt.gca().set(title='Position Holding Time Distribution', xlabel='Holding time days', ylabel='Frequency')
        # plt.show()

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
    def __init__(self, frequency, data, symbol, start_date, end_date, sma_short, sma_long, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data=data
        self.sma_short = sma_short
        self.sma_long = sma_long

    def set_parameters(self, SMA1=None, SMA2=None):
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['close'].rolling(self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['close'].rolling(self.SMA2).mean()

    def analyse_strategy(self, data):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='b', lw=.5)
        data["SMA1"].plot(ax=ax1, color='r', lw=.5)
        data["SMA2"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
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


class BollingerBandsBacktester(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, window_size, num_std_dev, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.window_size = window_size
        self.num_std_dev = num_std_dev

    def set_parameters(self, window_size=None, num_std_dev=None):
        ''' Updates Bollinger Bands parameters and respective time series. '''
        if window_size is not None:
            self.window_size = window_size
        if num_std_dev is not None:
            self.num_std_dev = num_std_dev

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='b', lw=.5)
        data["lower_band"].plot(ax=ax1, color='r', lw=.5)
        data["upper_band"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Lower Band", "Upper Band", "Buy", "Sell"])
        plt.title("Bollinger Bands Trading Strategy")
        # plt.show()

    def run_strategy(self):
        ''' Backtests the trading strategy. '''
        self.set_parameters(self.window_size, self.num_std_dev)
        data = self.data.copy().dropna()

        # Define the trading signals
        data['middle_band'] = data['close'].rolling(self.window_size).mean()
        data['std_dev'] = data['close'].rolling(self.window_size).std()
        data['upper_band'] = data['middle_band'] + (data['std_dev'] * self.num_std_dev)
        data['lower_band'] = data['middle_band'] - (data['std_dev'] * self.num_std_dev)
        data['position'] = np.where(data['close'] < data['lower_band'], 1, np.nan)  # buy signal
        data['position'] = np.where(data['close'] > data['upper_band'], -1, data['position'])  # sell signal
        data['position'] = data['position'].ffill().fillna(0)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        self.data_signal = data['position'][-1]
        self.analyse_strategy(data)
        return self.calculate_performance(data)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class RSIVectorBacktester(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, RSI_period, overbought_threshold, oversold_threshold,
                 amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.RSI_period = RSI_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold

    def set_parameters(self, RSI_period=None, overbought_threshold=None, oversold_threshold=None):
        if RSI_period is not None:
            self.RSI_period = RSI_period
        if overbought_threshold is not None:
            self.overbought_threshold = overbought_threshold
        if oversold_threshold is not None:
            self.oversold_threshold = oversold_threshold



    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='b', lw=.5)
        data["RSI"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "RSI", "Buy", "Sell"])
        plt.title("RSI Trading Strategy")
        # plt.show()

    def run_strategy(self):
        self.set_parameters(self.RSI_period, self.overbought_threshold, self.oversold_threshold)
        data = self.data.copy().dropna()

        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.RSI_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.RSI_period).mean()

        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Define trading signals
        data['position'] = np.where(data['RSI'] < self.oversold_threshold, 1, 0)  # buy signal
        data['position'] = np.where(data['RSI'] > self.overbought_threshold, -1, data['position'])  # sell signal

        # Implement the rest of the strategy logic
        self.data_signal = data['position'][-1]
        self.analyse_strategy(data)
        return self.calculate_performance(data)

    def generate_signal(self):
        self.run_strategy()
        return self.data_signal


class MomVectorBacktester(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, momentum, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data=data
        self.momentum = momentum

    def analyse_strategy(self, data):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
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
    def __init__(self, frequency, data, symbol, start_date, end_date, sma, threshold, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data=data
        self.sma=sma
        self.threshold=threshold

    def set_parameters(self, SMA=None):
        if SMA is not None:
            self.data['sma'] = self.data['close'].rolling(SMA).mean()

    def analyse_strategy(self, data):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='b', lw=.5)
        data["sma"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "SMA", "Buy", "Sell"])
        plt.title("Mean Reversion Trading Strategy")
        # plt.show()

    def run_strategy(self):
        self.set_parameters(int(self.sma))
        data = self.data.copy().dropna()
        data['distance'] = data['close'] - data['sma']
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
    def __init__(self, frequency, data, symbol, start_date, end_date, window_size, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data=data
        self.window_size=window_size


    def analyse_strategy(self, data):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data["close"].plot(ax=ax1, color='b', lw=.5)
        data["high"].plot(ax=ax1, color='g', lw=2.)
        data["low"].plot(ax=ax1, color='r', lw=2.)
        data["avg"].plot(ax=ax1, color='y', lw=2.)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Highs", "Lows", "Average", "Buy", "Sell"])
        plt.title("Turtle Trading Strategy")
        # plt.show()

    def run_strategy(self):
        data = self.data.copy().dropna()
        # window_size-days high
        data['high'] = data['close'].shift(1). \
        rolling(window=self.window_size).max()
        # window_size-days low
        data['low'] = data['close'].shift(1). \
        rolling(window=self.window_size).min()
        # window_size-days mean
        data['avg'] = data['close'].shift(1). \
        rolling(window=self.window_size).mean()
        data['long_entry'] = data['close'] > data.high
        data['short_entry'] = data['close'] < data.low
        data['long_exit'] = data['close'] < data.avg
        data['short_exit'] = data['close'] > data.avg
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

class ParabolicSARBacktester(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, SAR_step=0.02, SAR_max=0.2, amount=10000,
                 transaction_costs=0):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.SAR_step = SAR_step
        self.SAR_max = SAR_max

    def calculate_parabolic_sar(self):
        data = self.data.copy()
        data['SAR'] = data['close'].iloc[0]
        data['EP'] = data['close'].iloc[0]
        data['trend'] = 1
        data['AF'] = self.SAR_step

        for i in range(1, len(data)):
            if data['trend'][i - 1] == 1:  # Uptrend
                data['SAR'][i] = data['SAR'][i - 1] + data['AF'][i - 1] * (data['EP'][i - 1] - data['SAR'][i - 1])
                data['SAR'][i] = min(data['SAR'][i], data['close'].iloc[i - 1], data['close'].iloc[i - 2])

                if data['close'][i] > data['EP'][i - 1]:
                    data['EP'][i] = data['close'][i]
                    data['AF'][i] = min(data['AF'][i - 1] + self.SAR_step, self.SAR_max)
                else:
                    data['EP'][i] = data['EP'][i - 1]
                    data['AF'][i] = data['AF'][i - 1]

                if data['close'][i] < data['SAR'][i]:
                    data['trend'][i] = -1  # Switch to downtrend
                    data['SAR'][i] = data['EP'][i - 1]
                    data['EP'][i] = data['close'][i]
                    data['AF'][i] = self.SAR_step

            else:  # Downtrend
                data['SAR'][i] = data['SAR'][i - 1] + data['AF'][i - 1] * (data['EP'][i - 1] - data['SAR'][i - 1])
                data['SAR'][i] = max(data['SAR'][i], data['close'].iloc[i - 1], data['close'].iloc[i - 2])

                if data['close'][i] < data['EP'][i - 1]:
                    data['EP'][i] = data['close'][i]
                    data['AF'][i] = min(data['AF'][i - 1] + self.SAR_step, self.SAR_max)
                else:
                    data['EP'][i] = data['EP'][i - 1]
                    data['AF'][i] = data['AF'][i - 1]

                if data['close'][i] > data['SAR'][i]:
                    data['trend'][i] = 1  # Switch to uptrend
                    data['SAR'][i] = data['EP'][i - 1]
                    data['EP'][i] = data['close'][i]
                    data['AF'][i] = self.SAR_step

        return data

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='b', lw=.5)
        data["SAR"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "SAR", "Buy", "Sell"])
        plt.title("Parabolic SAR Trading Strategy")
        # plt.show()


    def run_strategy(self):
        ''' Backtests the trading strategy. '''
        data = self.calculate_parabolic_sar()

        # Define trading signals
        data['position'] = np.where(data['trend'] == 1, 1, -1)  # Long when trend is up, short when trend is down

        # Implement the rest of the strategy logic
        self.data_signal = data['position'][-1]
        self.analyse_strategy(data)  # Implement this method to visualize the strategy
        return self.calculate_performance(data)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal



class MACDStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, short_window, long_window, signal_window, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["macd"].plot(ax=ax1, color='g', lw=.5)
        data["signal"].plot(ax=ax1, color='r', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["macd"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["macd"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["MACD", "Signal", "Buy", "Sell"])
        plt.title("MACD Trading Strategy")
        # plt.show()


    def run_strategy(self):
        data = self.data.copy()
        # MACD calculation
        data['ema_short'] = data['close'].ewm(span=self.short_window, adjust=False).mean()
        data['ema_long'] = data['close'].ewm(span=self.long_window, adjust=False).mean()
        data['macd'] = data['ema_short'] - data['ema_long']
        data['signal'] = data['macd'].ewm(span=self.signal_window, adjust=False).mean()
        data['position'] = np.where(data['macd'] > data['signal'], 1, -1)
        self.data_signal = data['position'][-1]
        self.analyse_strategy(data)
        return self.calculate_performance(data)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class IchimokuStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, conversion_line_period=9, base_line_period=26,
                 leading_span_b_period=52, displacement=26, amount=10000, transaction_costs=0):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.conversion_line_period = conversion_line_period
        self.base_line_period = base_line_period
        self.leading_span_b_period = leading_span_b_period
        self.displacement = displacement


    def calculate_ichimoku(self):
        data = self.data.copy()

        # Conversion Line
        data['conversion_line'] = (data['high'].rolling(window=self.conversion_line_period).max() +
                                   data['low'].rolling(window=self.conversion_line_period).min()) / 2

        # Base Line
        data['base_line'] = (data['high'].rolling(window=self.base_line_period).max() +
                             data['low'].rolling(window=self.base_line_period).min()) / 2

        # Leading Span A
        data['leading_span_A'] = ((data['conversion_line'] + data['base_line']) / 2).shift(self.displacement)

        # Leading Span B
        data['leading_span_B'] = ((data['high'].rolling(window=self.leading_span_b_period).max() +
                                   data['low'].rolling(window=self.leading_span_b_period).min()) / 2).shift(self.displacement)

        # Lagging Span
        data['lagging_span'] = data['close'].shift(-self.displacement)

        return data

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='b', lw=.5)
        data["leading_span_A"].plot(ax=ax1, color='r', lw=0.5)
        data["leading_span_B"].plot(ax=ax1, color='g', lw=0.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Span A", "Span B", "Buy", "Sell"])
        plt.title("Ichimoku Trading Strategy")
        # plt.show()

    def run_strategy(self):
        data = self.calculate_ichimoku()

        # Trading signals based on Ichimoku
        # Example: Go long when the Conversion Line crosses above the Base Line
        # and the close is above the Cloud; go short when the opposite is true.
        data['position'] = np.where((data['conversion_line'] > data['base_line']) &
                                    (data['close'] > data['leading_span_A']) &
                                    (data['close'] > data['leading_span_B']), 1, 0)
        data['position'] = np.where((data['conversion_line'] < data['base_line']) &
                                    (data['close'] < data['leading_span_A']) &
                                    (data['close'] < data['leading_span_B']), -1, data['position'])
        self.data_signal = data['position'][-1]
        self.analyse_strategy(data)
        return self.calculate_performance(data)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class StochasticOscillatorStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, k_window=14, d_window=3, buy_threshold=20,
                 sell_threshold=80, amount=10000, transaction_costs=0):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.k_window = k_window
        self.d_window = d_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def calculate_stochastic_oscillator(self):
        data = self.data.copy()
        data['14-high'] = data['high'].rolling(window=self.k_window).max()
        data['14-low'] = data['low'].rolling(window=self.k_window).min()
        data['%K'] = 100 * ((data['close'] - data['14-low']) / (data['14-high'] - data['14-low']))
        data['%D'] = data['%K'].rolling(window=self.d_window).mean()
        return data

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data['close'].plot(ax=ax1, color='b', lw=.5)
        data["%K"].plot(ax=ax1, color='b', lw=.5)
        data['%D'].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["%K", "%D", "Buy", "Sell"])
        plt.title("Stochastic Oscillator Trading Strategy")
        # plt.show()

    def run_strategy(self):
        data = self.calculate_stochastic_oscillator()

        # Trading signals based on Stochastic Oscillator
        data['position'] = np.where(data['%K'] < data['%D'], 1, 0)  # Buy signal
        data['position'] = np.where(data['%K'] > data['%D'], -1, data['position'])  # Sell signal


        self.data_signal = data['position'][-1]
        self.analyse_strategy(data)  # Implement this method to visualize the strategy
        return self.calculate_performance(data)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class ADXStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, adx_period=14, di_period=14, threshold=25, amount=10000,
                 transaction_costs=0):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.adx_period = adx_period
        self.di_period = di_period
        self.threshold = threshold

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='b', lw=.5)
        data['+DI'].plot(ax=ax1, color='g', lw=.5)
        data['-DI'].plot(ax=ax1, color='r', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "+DI", "-DI", "Buy", "Sell"])
        plt.title("ADX Trading Strategy")
        # plt.show()

    def calculate_adx(self):
        data = self.data.copy()
        data['+DM'] = np.where((data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']),
                               data['high'] - data['high'].shift(1), 0)
        data['-DM'] = np.where((data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1)),
                               data['low'].shift(1) - data['low'], 0)

        data['+DM'] = np.where(data['+DM'] < 0, 0, data['+DM'])
        data['-DM'] = np.where(data['-DM'] < 0, 0, data['-DM'])

        data['TR'] = np.maximum((data['high'] - data['low']), np.maximum(abs(data['high'] - data['close'].shift(1)),
                                                                         abs(data['low'] - data['close'].shift(1))))

        data['+DI'] = 100 * data['+DM'].rolling(window=self.di_period).sum() / data['TR'].rolling(
            window=self.di_period).sum()
        data['-DI'] = 100 * data['-DM'].rolling(window=self.di_period).sum() / data['TR'].rolling(
            window=self.di_period).sum()

        data['DX'] = 100 * abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])
        data['ADX'] = data['DX'].rolling(window=self.adx_period).mean()

        return data

    def run_strategy(self):
        data = self.calculate_adx()

        # Trading signals based on ADX, +DI, and -DI
        data['position'] = np.where((data['+DI'] > data['-DI']) & (data['ADX'] > self.threshold), 1, 0)  # Buy signal
        data['position'] = np.where((data['-DI'] > data['+DI']) & (data['ADX'] > self.threshold), -1,
                                    data['position'])  # Sell signal

        # Implement the rest of the strategy logic
        self.data_signal = data['position'][-1]
        self.analyse_strategy(data)  # Implement this method to visualize the strategy
        return self.calculate_performance(data)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal


class VolumeStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, volume_threshold, volume_window, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.volume_threshold = volume_threshold
        self.volume_window=volume_window

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["volume"].plot(ax=ax1, color='g', lw=.5)
        data["average_volume"].plot(ax=ax1, color='r', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["volume"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["volume"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Volume", "Average Volume", "Buy", "Sell"])
        plt.title("Volume Trading Strategy")
        # plt.show()


    def run_strategy(self):
        data = self.data.copy()

        # Calculate the average volume over a specified period
        data['average_volume'] = data['volume'].rolling(window=self.volume_window).mean()

        # Identify significant volume increases - a volume spike
        data['volume_spike'] = data['volume'] > (data['average_volume'] * self.volume_threshold)

        # Define a simple trading logic: buy when there is a volume spike, sell otherwise
        data['position'] = np.where(data['volume_spike'], 1, -1)
        self.data_signal = data['position'][-1]
        self.analyse_strategy(data)
        return self.calculate_performance(data)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal


class WilliamsRBacktester(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, lookback_period, overbought, oversold, amount,
                 transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.lookback_period = lookback_period
        self.overbought = overbought
        self.oversold = oversold


    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Buy", "Sell"])
        plt.title("William R Trading Strategy")
        # plt.show()

    def run_strategy(self):
        data=self.data.copy()
        high = data['high'].rolling(self.lookback_period).max()
        low = data['low'].rolling(self.lookback_period).min()
        data['%R'] = -100 * (high - data['close']) / (high - low)

        data['position'] = np.where(data['%R'] < -self.oversold, 1, 0)
        data['position'] = np.where(data['%R'] > -self.overbought, -1, data['position'])
        self.data_signal = data['position'][-1]
        self.analyse_strategy(data)
        return self.calculate_performance(data)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal


class VolatilityBreakoutBacktester(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, volatility_window, breakout_factor, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.volatility_window = volatility_window
        self.breakout_factor = breakout_factor
        self.set_parameters(volatility_window, breakout_factor)

    def set_parameters(self, volatility_window=None, breakout_factor=None):
        if volatility_window is not None:
            self.volatility_window = volatility_window
        if breakout_factor is not None:
            self.breakout_factor = breakout_factor



    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data=data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='b', lw=.5)
        data["lower_band"].plot(ax=ax1, color='r', lw=.5)
        data["upper_band"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Lower Band", "Upper Band", "Buy", "Sell"])
        plt.title("Volatility Breakout Trading Strategy")
        # plt.show()

    def run_strategy(self):
        ''' Backtests the trading strategy. '''
        data = self.data.copy().dropna()

        # Calculate ATR and bands
        data['ATR'] = self.data['close'].rolling(self.volatility_window).std()
        data['upper_band'] = data['close'] + data['ATR'] * self.breakout_factor
        data['lower_band'] = data['close'] - data['ATR'] * self.breakout_factor

        # Define trading signals
        data['position'] = np.where(data['close'] > data['upper_band'], 1, np.nan)
        data['position'] = np.where(data['close'] < data['lower_band'], -1, data['position'])
        data['position'] = data['position'].ffill().fillna(0)

        self.data_signal = data['position'][-1]
        self.analyse_strategy(data)
        return self.calculate_performance(data)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal


