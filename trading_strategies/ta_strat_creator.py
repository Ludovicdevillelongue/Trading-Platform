import numpy as np
from matplotlib import pyplot as plt
import talib
from trading_strategies.indicator_strat_creator import StrategyCreator
from signal_generator.ml_predictor import MLPredictor


class BBandsStrategy(StrategyCreator):
    def __init__(self,strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, timeperiod, nbdevup,
                 nbdevdn, reg_method, amount, transaction_costs,predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
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

    def run_strategy(self, predictive_strat=False):
        data_strat = self.data.copy()
        data_strat['upper_band'], data_strat['middle_band'], data_strat['lower_band'] = talib.BBANDS(
            data_strat['close'], timeperiod=self.timeperiod,
            nbdevup=self.nbdevup, nbdevdn=self.nbdevdn, matype=0)

        # Buy when the close prices cross below the lower band, and sell when they cross above the upper band
        data_strat['position'] = np.where(data_strat['close'] < data_strat['lower_band'], 1,
                                    np.where(data_strat['close'] > data_strat['upper_band'], self.strat_type_pos, 0))
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['close', 'lower_band', 'upper_band'], 1).run()
            data_sized = self.regression_positions(data_pred, "close", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "close", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class MAStrategy(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, short_period, long_period,
                 reg_method, amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.short_period = short_period
        self.long_period = long_period
        self.reg_method = reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data = data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Buy", "Sell"])
        plt.title("MA Crossover Trading Strategy")

    def run_strategy(self, predictive_strat=False):
        data_strat = self.data.copy()
        data_strat['short_ma'] = talib.SMA(data_strat['close'], timeperiod=self.short_period)
        data_strat['long_ma'] = talib.SMA(data_strat['close'], timeperiod=self.long_period)

        # Generate signals based on MA crossover
        data_strat['position'] = np.where(data_strat['short_ma'] > data_strat['long_ma'], 1,
                                          np.where(data_strat['short_ma'] < data_strat['long_ma'], self.strat_type_pos, 0))
        data_strat = data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['close', 'short_ma', 'long_ma'], 1).run()
            data_sized = self.regression_positions(data_pred, "close", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "close", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class DEMAStrategy(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, short_period, long_period,
                 reg_method, amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.short_period = short_period
        self.long_period = long_period
        self.reg_method = reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data = data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Buy", "Sell"])
        plt.title("DEMA Crossover Trading Strategy")

    def run_strategy(self, predictive_strat=False):
        data_strat = self.data.copy()
        data_strat['short_dema'] = talib.DEMA(data_strat['close'], timeperiod=self.short_period)
        data_strat['long_dema'] = talib.DEMA(data_strat['close'], timeperiod=self.long_period)

        # Generate signals based on DEMA crossover
        data_strat['position'] = np.where(data_strat['short_dema'] > data_strat['long_dema'], 1,
                                          np.where(data_strat['short_dema'] < data_strat['long_dema'], self.strat_type_pos, 0))
        data_strat = data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['close', 'short_dema', 'long_dema'], 1).run()
            data_sized = self.regression_positions(data_pred, "close", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "close", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal

class EMAStrategy(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, short_period, long_period,
                 reg_method, amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.short_period = short_period
        self.long_period = long_period
        self.reg_method = reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data = data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Buy", "Sell"])
        plt.title("EMA Crossover Trading Strategy")

    def run_strategy(self, predictive_strat=False):
        data_strat = self.data.copy()
        data_strat['short_ema'] = talib.EMA(data_strat['close'], timeperiod=self.short_period)
        data_strat['long_ema'] = talib.EMA(data_strat['close'], timeperiod=self.long_period)

        # Generate signals based on EMA crossover
        data_strat['position'] = np.where(data_strat['short_ema'] > data_strat['long_ema'], 1,
                                          np.where(data_strat['short_ema'] < data_strat['long_ema'], self.strat_type_pos, 0))
        data_strat = data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['close', 'short_ema', 'long_ema'], 1).run()
            data_sized = self.regression_positions(data_pred, "close", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "close", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class HTTrendlineStrategy(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, period, reg_method,
                 amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.period = period
        self.reg_method = reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data = data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Buy", "Sell"])
        plt.title("HT Trendline Strategy")

    def run_strategy(self, predictive_strat=False):
        data_strat = self.data.copy()
        data_strat['ht_trendline'] = talib.HT_TRENDLINE(data_strat['close'])

        # Generate signals based on HT Trendline
        data_strat['position'] = np.where(data_strat['close'] > data_strat['ht_trendline'], 1,
                                          np.where(data_strat['close'] < data_strat['ht_trendline'], self.strat_type_pos, 0))
        data_strat = data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['close', 'ht_trendline'], 1).run()
            data_sized = self.regression_positions(data_pred, "close", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "close", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class OBVStrategy(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, reg_method,
                 amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.reg_method = reg_method

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data = data.dropna(axis=0)
        data["close"].plot(ax=ax1, color='g', lw=.5)
        ax1.plot(data.loc[data.orders >= 1.0].index,
                 data["close"][data.orders >= 1.0],
                 '^', markersize=7, color='k')
        ax1.plot(data.loc[data.orders <= -1.0].index,
                 data["close"][data.orders <= -1.0],
                 'v', markersize=7, color='k')
        plt.legend(["Price", "Buy", "Sell"])
        plt.title("OBV Strategy")

    def run_strategy(self, predictive_strat=False):
        data_strat = self.data.copy()
        data_strat['obv'] = talib.OBV(data_strat['close'], data_strat['volume'])

        # Generate signals based on OBV
        data_strat['position'] = np.where(data_strat['obv'] > 0, 1, -1)
        data_strat = data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['close', 'obv'], 1).run()
            data_sized = self.regression_positions(data_pred, "close", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "close", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal