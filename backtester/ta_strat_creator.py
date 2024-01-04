import numpy as np
from matplotlib import pyplot as plt
import talib
from backtester.indicator_strat_creator import StrategyCreator

class BBandsStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, timeperiod, nbdevup, nbdevdn, reg_method,
                 amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.timeperiod = timeperiod
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn
        self.reg_method=reg_method

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
        plt.title("BB Trading Strategy")
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['upper_band'], data_strat['middle_band'], data_strat['lower_band'] = talib.BBANDS(
            data_strat['close'], timeperiod=self.timeperiod,
            nbdevup=self.nbdevup, nbdevdn=self.nbdevdn, matype=0)

        # Buy when the close prices cross below the lower band, and sell when they cross above the upper band
        data_strat['position'] = np.where(data_strat['close'] < data_strat['lower_band'], 1,
                                    np.where(data_strat['close'] > data_strat['upper_band'], -1, 0))
        data_strat=data_strat.dropna(axis=0)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class DEMAStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, timeperiod, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.timeperiod = timeperiod
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and DEMA
        data["close"].plot(ax=ax, color='g', lw=1)
        data["dema"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k',
                label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k',
                label='Sell Signal')

        plt.title("DEMA Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['dema'] = talib.DEMA(data_strat['close'], timeperiod=self.timeperiod)

        # Strategy: Buy when the close is above the DEMA, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['dema'], 1, -1)
        data_strat=data_strat.dropna(axis=0)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class EMAStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, timeperiod, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.timeperiod = timeperiod
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and EMA
        data["close"].plot(ax=ax, color='g', lw=1)
        data["ema"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k',
                label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k',
                label='Sell Signal')

        plt.title("EMA Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['ema'] = talib.EMA(data_strat['close'], timeperiod=self.timeperiod)

        # Strategy: Buy when the close is above the EMA, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['ema'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)
    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class HTTrendlineStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and HT Trendline
        data["close"].plot(ax=ax, color='g', lw=1)
        data["ht_trendline"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("HT Trendline Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['ht_trendline'] = talib.HT_TRENDLINE(data_strat['close'])

        # Strategy: Buy when the close is above the HT Trendline, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['ht_trendline'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class KAMAStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, timeperiod, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.timeperiod = timeperiod
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and KAMA
        data["close"].plot(ax=ax, color='g', lw=1)
        data["kama"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("KAMA Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['kama'] = talib.KAMA(data_strat['close'], timeperiod=self.timeperiod)

        # Strategy: Buy when the close is above the KAMA, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['kama'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal


class MAStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, timeperiod, ma_type,reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.timeperiod = timeperiod
        self.ma_type = ma_type  # MA type (e.g., SIMPLE, EXPONENTIAL)
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and MA
        data["close"].plot(ax=ax, color='g', lw=1)
        data["ma"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("MA Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()

        if self.ma_type == 'SIMPLE':
            data_strat['ma'] = talib.SMA(data_strat['close'], timeperiod=self.timeperiod)
        elif self.ma_type == 'EXPONENTIAL':
            data_strat['ma'] = talib.EMA(data_strat['close'], timeperiod=self.timeperiod)
        else:
            raise ValueError("Invalid MA Type")

        # Strategy: Buy when the close is above the MA, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['ma'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class MAMAStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, fastlimit, slowlimit, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.fastlimit = fastlimit
        self.slowlimit = slowlimit
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price, MAMA and FAMA
        data["close"].plot(ax=ax, color='g', lw=1)
        data["mama"].plot(ax=ax, color='b', lw=1)
        data["fama"].plot(ax=ax, color='r', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("MAMA Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['mama'], data_strat['fama'] = talib.MAMA(data_strat['close'], fastlimit=self.fastlimit, slowlimit=self.slowlimit)

        # Strategy: Buy when MAMA crosses above FAMA, sell when MAMA crosses below FAMA
        data_strat['position'] = np.where(data_strat['mama'] > data_strat['fama'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class MAVPStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, periods, minperiod, maxperiod, reg_method,
                 amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.periods = periods  # This should be an array or Series of periods
        self.minperiod = minperiod
        self.maxperiod = maxperiod
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and MAVP
        data["close"].plot(ax=ax, color='g', lw=1)
        data["mavp"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("MAVP Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['mavp'] = talib.MAVP(data_strat['close'], self.periods, minperiod=self.minperiod, maxperiod=self.maxperiod)

        # Strategy: Define your buy/sell conditions based on MAVP
        # Example: Buy when close is greater than MAVP, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['mavp'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class MIDPOINTStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, timeperiod, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.timeperiod = timeperiod  # Time period for calculating the midpoint
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and MIDPOINT
        data["close"].plot(ax=ax, color='g', lw=1)
        data["midpoint"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("MIDPOINT Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['midpoint'] = talib.MIDPOINT(data_strat['close'], timeperiod=self.timeperiod)

        # Strategy: Define your buy/sell conditions based on MIDPOINT
        # Example: Buy when close is above MIDPOINT, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['midpoint'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class MIDPRICEStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, timeperiod, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.timeperiod = timeperiod  # Time period for calculating the MIDPRICE
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and MIDPRICE
        data["close"].plot(ax=ax, color='g', lw=1)
        data["midprice"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("MIDPRICE Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['midprice'] = talib.MIDPRICE(data_strat['high'], data_strat['low'], timeperiod=self.timeperiod)

        # Strategy: Define your buy/sell conditions based on MIDPRICE
        # Example: Buy when close is above MIDPRICE, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['midprice'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class SARStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, acceleration, maximum, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.acceleration = acceleration
        self.maximum = maximum
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and SAR
        data["close"].plot(ax=ax, color='g', lw=1)
        ax.scatter(data.index, data['sar'], color='b', label='SAR')

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("SAR Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['sar'] = talib.SAR(data_strat['high'], data_strat['low'], acceleration=self.acceleration, maximum=self.maximum)

        # Strategy: Buy when the close is above the SAR, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['sar'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class SAREXTStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, start_value, offset_on_reverse,
                 acceleration_init_long, acceleration_long, acceleration_max_long, acceleration_init_short,
                 acceleration_short, acceleration_max_short, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.start_value = start_value
        self.offset_on_reverse = offset_on_reverse
        self.acceleration_init_long = acceleration_init_long
        self.acceleration_long = acceleration_long
        self.acceleration_max_long = acceleration_max_long
        self.acceleration_init_short = acceleration_init_short
        self.acceleration_short = acceleration_short
        self.acceleration_max_short = acceleration_max_short
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and SAREXT
        data["close"].plot(ax=ax, color='g', lw=1)
        data["sarext"].plot(ax=ax, color='b', lw=1, linestyle='--')

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("SAREXT Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['sarext'] = talib.SAREXT(data_strat['high'], data_strat['low'], startvalue=self.start_value,
                                      offsetonreverse=self.offset_on_reverse,
                                      accelerationinitlong=self.acceleration_init_long,
                                      accelerationlong=self.acceleration_long,
                                      accelerationmaxlong=self.acceleration_max_long,
                                      accelerationinitshort=self.acceleration_init_short,
                                      accelerationshort=self.acceleration_short,
                                      accelerationmaxshort=self.acceleration_max_short)

        # Strategy: Define your buy/sell conditions based on SAREXT
        # Example: Buy when the close is above SAREXT, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['sarext'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class SMAStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, timeperiod, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.timeperiod = timeperiod  # Time period for calculating the SMA
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and SMA
        data["close"].plot(ax=ax, color='g', lw=1)
        data["sma"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("SMA Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['sma'] = talib.SMA(data_strat['close'], timeperiod=self.timeperiod)

        # Strategy: Define your buy/sell conditions based on SMA
        # Example: Buy when close is above SMA, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['sma'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class T3Strategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, timeperiod, volume_factor, reg_method,
                 amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.timeperiod = timeperiod  # Time period for calculating the T3
        self.volume_factor = volume_factor  # Volume factor for the T3 calculation
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and T3
        data["close"].plot(ax=ax, color='g', lw=1)
        data["t3"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("T3 Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['t3'] = talib.T3(data_strat['close'], timeperiod=self.timeperiod, vfactor=self.volume_factor)

        # Strategy: Define your buy/sell conditions based on T3
        # Example: Buy when close is above T3, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['t3'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class TEMAStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, timeperiod, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.timeperiod = timeperiod  # Time period for calculating the TEMA
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and TEMA
        data["close"].plot(ax=ax, color='g', lw=1)
        data["tema"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("TEMA Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['tema'] = talib.TEMA(data_strat['close'], timeperiod=self.timeperiod)

        # Strategy: Define your buy/sell conditions based on TEMA
        # Example: Buy when close is above TEMA, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['tema'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class TRIMAStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, timeperiod, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.timeperiod = timeperiod  # Time period for calculating the TRIMA
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and TRIMA
        data["close"].plot(ax=ax, color='g', lw=1)
        data["trima"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("TRIMA Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['trima'] = talib.TRIMA(data_strat['close'], timeperiod=self.timeperiod)

        # Strategy: Define your buy/sell conditions based on TRIMA
        # Example: Buy when close is above TRIMA, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['trima'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class WMAStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, timeperiod, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.timeperiod = timeperiod  # Time period for calculating the WMA
        self.reg_method=reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and WMA
        data["close"].plot(ax=ax, color='g', lw=1)
        data["wma"].plot(ax=ax, color='b', lw=1)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("WMA Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['wma'] = talib.WMA(data_strat['close'], timeperiod=self.timeperiod)

        # Strategy: Define your buy/sell conditions based on WMA
        # Example: Buy when close is above WMA, sell when below
        data_strat['position'] = np.where(data_strat['close'] > data_strat['wma'], 1, -1)

        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        data_sized=self.regression_positions(data_strat, "close", self.reg_method)
        self.data_signal=data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal


class ADStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.reg_method = reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and AD line
        data["close"].plot(ax=ax, color='g', lw=1)
        data["ad_line"].plot(ax=ax, color='b', lw=1, secondary_y=True)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("AD Line Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['ad_line'] = talib.AD(data_strat['high'], data_strat['low'], data_strat['close'], data_strat['volume'])

        # Strategy: Define your buy/sell conditions based on AD line
        # Example: Buy when AD line is increasing, sell when decreasing
        data_strat['position'] = np.where(data_strat['ad_line'].diff() > 0, 1, -1)

        data_sized = self.regression_positions(data_strat, "ad_line", self.reg_method)
        self.data_signal = data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal


class ADOSCStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, fast_period, slow_period, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.fast_period = fast_period  # Fast period for ADOSC calculation
        self.slow_period = slow_period  # Slow period for ADOSC calculation
        self.reg_method = reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and ADOSC
        data["close"].plot(ax=ax, color='g', lw=1)
        data["adosc"].plot(ax=ax, color='b', lw=1, secondary_y=True)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("ADOSC Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['adosc'] = talib.ADOSC(data_strat['high'], data_strat['low'], data_strat['close'], data_strat['volume'], fastperiod=self.fast_period, slowperiod=self.slow_period)

        # Strategy: Define your buy/sell conditions based on ADOSC
        # Example: Buy when ADOSC is increasing, sell when decreasing
        data_strat['position'] = np.where(data_strat['adosc'].diff() > 0, 1, -1)
        data_strat=data_strat.dropna(axis=0)
        data_sized = self.regression_positions(data_strat, "adosc", self.reg_method)
        self.data_signal = data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal

class OBVStrategy(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data = data
        self.reg_method = reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel(f'{self.symbol} price in $')

        data['orders'] = data['position'].diff()
        data = data.dropna()

        # Plot close price and OBV
        data["close"].plot(ax=ax, color='g', lw=1)
        data["obv"].plot(ax=ax, color='b', lw=1, secondary_y=True)

        # Plot Buy and Sell signals
        ax.plot(data.loc[data.orders > 0].index, data["close"][data.orders > 0], '^', markersize=10, color='k', label='Buy Signal')
        ax.plot(data.loc[data.orders < 0].index, data["close"][data.orders < 0], 'v', markersize=10, color='k', label='Sell Signal')

        plt.title("OBV Trading Strategy")
        plt.legend()
        # plt.show()

    def run_strategy(self):
        data_strat = self.data.copy()
        data_strat['obv'] = talib.OBV(data_strat['close'], data_strat['volume'])

        # Strategy: Define your buy/sell conditions based on OBV
        # Example: Buy when OBV is increasing, sell when decreasing
        data_strat['position'] = np.where(data_strat['obv'].diff() > 0, 1, -1)

        data_sized = self.regression_positions(data_strat, "obv", self.reg_method)
        self.data_signal = data_sized['regularized_position'][-1]
        self.analyse_strategy(data_sized)
        return self.calculate_performance(data_sized)

    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        self.run_strategy()
        return self.data_signal