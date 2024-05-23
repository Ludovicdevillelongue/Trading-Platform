import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from indicators.performances_indicators import (SharpeRatio, SortinoRatio, MaxDrawdown,
                        CalmarRatio, Alpha,
                        Beta, LogReturns, Returns, CumulativeReturns, CumulativeLogReturns)
from single_asset_trader.signal_generator.ml_predictor import MLPredictor
import warnings

# To deactivate all warnings:
warnings.filterwarnings('ignore')


# A base backtesting class with common functionality
class StrategyCreator:
    def __init__(self, strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                 predictive_strat):
        self.strat_type_pos=strat_type_pos
        self.frequency = frequency
        self.symbol = symbol
        self.risk_free_rate = risk_free_rate
        self.start_date = start_date
        self.end_date = end_date
        self.amount = amount
        self.transaction_costs = transaction_costs
        self.predictive_strat = predictive_strat
        self.data = None  # Will hold historical price data
        self.results = None  # Will hold backtest results

    def calculate_returns(self, downloaded_data):
        ''' Retrieves and prepares the data'''
        raw=downloaded_data.copy()
        raw['returns']=Returns().get_metric(raw['close'])
        raw['log_returns']=LogReturns().get_metric(raw['returns'])
        raw['creturns']=CumulativeReturns().get_metric(1, raw['returns'])
        raw['log_creturns']=CumulativeLogReturns().get_metric(1, raw['log_returns'])
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
            data[col] = data['returns'].shift(lag)
            self.cols.append(col)
        data.dropna(inplace=True)  # Drop rows with NaN values
        self.lagged_data = data

    def fit_model(self, start, end, lags, model):
        ''' Implements the fitting step.
        '''

        self.prepare_lags(start, end, lags)
        if model == 'linalg':
            self.reg = np.linalg.lstsq(self.lagged_data[self.cols],
                                       np.sign(self.lagged_data['returns']),
                                       rcond=None)[0]
        else:
            model.fit(self.lagged_data[self.cols],
                      np.sign(self.lagged_data['returns']))

    def prepare_lags_for_signal(self, lags):
        ''' Prepares the lagged data for the most recent data point for signal generation. '''
        latest_data = self.data.tail(2 * lags + 1).copy()  # Get enough rows for lags
        cols = [f'lag_{lag}' for lag in range(1, lags + 1)]
        for lag in range(1, lags + 1):
            latest_data[f'lag_{lag}'] = latest_data['returns'].shift(lag)
        latest_data.dropna(inplace=True)
        return latest_data, cols

    def fit_model_for_signal(self, lags, model):
        latest_data, cols = self.prepare_lags_for_signal(lags)
        latest_row = latest_data.tail(1)
        if model == 'linalg':
            self.reg = np.linalg.lstsq(latest_row[cols],
                                       np.sign(latest_row['returns']),
                                       rcond=None)[0]
            return latest_row
        else:
            model.fit(latest_row[cols], np.sign(latest_row['returns']))
            return latest_row

    def calculate_accuracy_metrics(self, data):
        y_true = data['position']  # Actual values
        y_pred = data['sized_position']  # Predicted values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        # RMSE or MAE less than 0.1 might be considered good
        return {'RMSE': rmse, 'MAE': mae}

    def regression_positions(self, data_strat, signal_metric, position_type, reg_method):
        data_regression = data_strat[[signal_metric, position_type]]
        try:
            X = data_regression[[signal_metric]].values.reshape(-1, 1)
            y = data_regression[position_type].values

            if reg_method == "linear":
                model = LinearRegression()

            elif reg_method == "poly":
                poly = PolynomialFeatures(degree=2)
                X = poly.fit_transform(X)
                model = LinearRegression()

            elif reg_method == "logistic":
                model = LogisticRegression()
                y = (y > 0).astype(int)

            elif reg_method == "ridge":
                model = self.__tune_ridge(X, y)

            elif reg_method == "lasso":
                model = self.__tune_lasso(X, y)

            elif reg_method == "elastic_net":
                model = self.__tune_elastic_net(X, y)

            elif reg_method == "bayesian":
                model = self.__tune_bayesian_ridge(X, y)

            elif reg_method == "svr":
                model = self.__train_svr_model(X, y)

            elif reg_method == "no_regression":
                data_regression["sized_position"] = y
                return self.__finalize_data(data_strat, data_regression)

            model.fit(X, y)
            predictions = model.predict(X)
            if self.strat_type_pos==0:
                # Apply the constraint: Ensure that predictions do not go below 0
                predictions = np.maximum(predictions, 0)
            else:
                pass
            data_regression["sized_position"] = predictions
            data_regression.loc[data_regression[position_type] == 0, 'sized_position'] = 0
            return self.__finalize_data(data_strat, data_regression)

        except Exception as e:
            data_regression["sized_position"] = data_regression['position']
            return self.__finalize_data(data_strat, data_regression)

    def __tune_ridge(self, X, y):
        param_grid = {'alpha': [1e-3, 1e-2, 1e-1, 1]}
        grid_search = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def __tune_lasso(self, X, y):
        pipeline = make_pipeline(StandardScaler(), Lasso())
        param_grid = {'lasso__alpha': [1e-3, 1e-2, 1e-1, 1]}
        grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def __tune_elastic_net(self, X, y):
        pipeline = make_pipeline(StandardScaler(), ElasticNet())
        param_grid = {'elasticnet__alpha': [1e-3, 1e-2, 1e-1, 1], 'elasticnet__l1_ratio': [0.2, 0.5, 0.8]}
        grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def __tune_bayesian_ridge(self, X, y):
        param_grid = {
            'alpha_1': [1e-6, 1e-5, 1e-4],
            'alpha_2': [1e-6, 1e-5, 1e-4],
            'lambda_1': [1e-6, 1e-5, 1e-4],
            'lambda_2': [1e-6, 1e-5, 1e-4]
        }
        grid_search = GridSearchCV(BayesianRidge(), param_grid, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def __train_svr_model(self, X, y):
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
        param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['rbf']}
        grid_search = GridSearchCV(SVR(), param_grid, scoring='neg_mean_squared_error')
        grid_search.fit(X_scaled, y_scaled.ravel())
        best_model = grid_search.best_estimator_
        return make_pipeline(scaler_X, best_model)

    def __finalize_data(self, data_strat, data_regression):
        data_sized = pd.concat((data_strat, data_regression[['sized_position']]), axis=1)
        data_sized['sized_position'] = data_sized['sized_position'].fillna(0)
        self.calculate_accuracy_metrics(data_sized)
        data_sized = self.regularize_sizing(data_sized, 0.5)
        return data_sized

    def regularize_sizing(self, data_sized, percent_change):
        data_sized['percentage_change'] = data_sized['sized_position'].pct_change()
        mask = data_sized['percentage_change'].abs() > percent_change
        data_sized['regularized_position'] = data_sized['sized_position'].where(mask, other=pd.NA).ffill()
        data_sized.at[data_sized.index[0], 'regularized_position'] = data_sized['sized_position'].iloc[0]
        data_sized['regularized_position'] = data_sized['regularized_position'].fillna(method='ffill')
        return data_sized

    def calculate_performance(self, data):
        """ Calculate performance and return a DataFrame with results. """
        # get relevant columns
        data = data[['returns', 'creturns', 'log_returns', 'log_creturns', 'regularized_position', 'orders']]
        # Calculate strategy return
        data['strategy'] = data['regularized_position'].shift(1) * data['returns']
        data['log_strategy']=data['regularized_position'].shift(1) * data['log_returns']
        data['strategy'].iloc[0] = 0
        data['orders'].iloc[0] = data['regularized_position'].iloc[0]

        # Subtract transaction costs from return when trade takes place
        data.loc[data['orders'] != 0, 'strategy'] -= self.transaction_costs * abs(data['orders'])

        # Calculate cumulative returns of the strategy
        data['cstrategy']=CumulativeReturns().get_metric(1, data['strategy'])

        # Save results for further output or plot
        self.results = data

        # Calculate absolute and out-/underperformance
        aperf = self.results['cstrategy'].iloc[-1]
        operf = aperf - data['creturns'].iloc[-1]

        # Calculate Performance Indicators
        risk_free_rate = self.risk_free_rate

        try:
            if data['regularized_position'].eq(0).all():
                sharpe_ratio = 0
            else:
                sharpe_ratio = SharpeRatio(self.frequency, risk_free_rate).calculate(data['strategy'])
        except Exception as e:
            sharpe_ratio = 0

        sortino_ratio = SortinoRatio(self.frequency, 0).calculate(data['strategy'])
        max_drawdown = MaxDrawdown().calculate(data['cstrategy'])
        calmar_ratio = CalmarRatio(self.frequency).calculate(data['strategy'], max_drawdown)

        try:
            beta = Beta().calculate(data['strategy'], data['returns'])
            alpha = Alpha(self.frequency, risk_free_rate).calculate(data['strategy'], data['returns'], beta)
        except Exception as e:
            beta = 0
            alpha = 0
        return round(aperf, 0), round(operf, 0), round(sharpe_ratio, 2), round(sortino_ratio, 2), round(calmar_ratio,2), \
               round(max_drawdown, 0), round(alpha, 2), round(beta, 2)

    def position_holding_time(self, data):
        position_holding_times = []
        current_pos = 0
        current_pos_start = 0
        for i in range(0, len(data)):
            pos = data['regularized_position'].iloc[i]
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
    def __init__(self,strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, sma_short, sma_long, reg_method,
                 amount, transaction_costs,
                 predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.reg_method = reg_method

    def set_parameters(self, sma_short=None, sma_long=None):
        if sma_short is not None:
            self.sma1= self.data['close'].rolling(sma_short).mean()
        if sma_long is not None:
            self.sma2 = self.data['close'].rolling(sma_long).mean()

    def analyse_strategy(self, data):
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data["close"].plot(ax=ax1, color='b', lw=.5)
        # data["SMA1"].plot(ax=ax1, color='r', lw=.5)
        # data["SMA2"].plot(ax=ax1, color='g', lw=.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["close"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["close"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["Price", "Short mavg", "Long mavg", "Buy", "Sell"])
        # plt.title("Simple Moving Average Trading Strategy")

    def run_strategy(self, predictive_strat=False):

        '''
        Backtests the trading strategy.
        '''
        self.set_parameters(int(self.sma_short), int(self.sma_long))
        data_strat = self.data.copy().dropna()
        data_strat['SMA1']=self.sma1
        data_strat['SMA2']=self.sma2
        data_strat['position'] = np.where(data_strat['SMA1'] > data_strat['SMA2'], 1, self.strat_type_pos)
        data_strat['diff_SMA'] = data_strat['SMA1'] - data_strat['SMA2']
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['SMA1', 'SMA2'], 1).run()
            data_sized = self.regression_positions(data_pred, "diff_SMA", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "diff_SMA", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class BollingerBandsBacktester(StrategyCreator):
    def __init__(self,strat_type_pos, frequency, data, symbol, risk_free_rate,
                 start_date, end_date, window_size, num_std_dev, reg_method, amount, transaction_costs,
                 predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.window_size = window_size
        self.num_std_dev = num_std_dev
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat

    def set_parameters(self, window_size=None, num_std_dev=None):
        ''' Updates Bollinger Bands parameters and respective time series. '''
        if window_size is not None:
            self.window_size = window_size
        if num_std_dev is not None:
            self.num_std_dev = num_std_dev

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data["close"].plot(ax=ax1, color='b', lw=.5)
        # data["lower_band"].plot(ax=ax1, color='r', lw=.5)
        # data["upper_band"].plot(ax=ax1, color='g', lw=.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["close"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["close"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["Price", "Lower Band", "Upper Band", "Buy", "Sell"])
        # plt.title("Bollinger Bands Trading Strategy")
        # plt.show()

    def run_strategy(self, predictive_strat=False):
        ''' Backtests the trading strategy. '''
        self.set_parameters(self.window_size, self.num_std_dev)
        data_strat = self.data.copy().dropna()

        # Define the trading signals
        data_strat['middle_band'] = data_strat['close'].rolling(self.window_size).mean()
        data_strat['std_dev'] = data_strat['close'].rolling(self.window_size).std()
        data_strat['upper_band'] = data_strat['middle_band'] + (data_strat['std_dev'] * self.num_std_dev)
        data_strat['lower_band'] = data_strat['middle_band'] - (data_strat['std_dev'] * self.num_std_dev)
        data_strat['position'] = np.where(data_strat['close'] < data_strat['lower_band'], 1, np.nan)  # buy signal
        data_strat['position'] = np.where(data_strat['close'] > data_strat['upper_band'], self.strat_type_pos,
                                          data_strat['position'])  # sell signal
        data_strat['position'] = data_strat['position'].ffill().fillna(0)
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


class RSIVectorBacktester(StrategyCreator):
    def __init__(self,strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, RSI_period, overbought_threshold,
                 oversold_threshold,
                 reg_method, amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.RSI_period = RSI_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat

    def set_parameters(self, RSI_period=None, overbought_threshold=None, oversold_threshold=None):
        if RSI_period is not None:
            self.RSI_period = RSI_period
        if overbought_threshold is not None:
            self.overbought_threshold = overbought_threshold
        if oversold_threshold is not None:
            self.oversold_threshold = oversold_threshold

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data["close"].plot(ax=ax1, color='b', lw=.5)
        # data["RSI"].plot(ax=ax1, color='g', lw=.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["close"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["close"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["Price", "RSI", "Buy", "Sell"])
        # plt.title("RSI Trading Strategy")
        # plt.show()

    def run_strategy(self, predictive_strat=False):
        self.set_parameters(self.RSI_period, self.overbought_threshold, self.oversold_threshold)
        data_strat = self.data.copy().dropna()

        # Calculate RSI
        delta = data_strat['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.RSI_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.RSI_period).mean()

        rs = gain / loss
        data_strat['RSI'] = 100 - (100 / (1 + rs))

        # Define trading signals
        data_strat['position'] = np.where(data_strat['RSI'] < self.oversold_threshold, 1, 0)  # buy signal
        data_strat['position'] = np.where(data_strat['RSI'] > self.overbought_threshold, self.strat_type_pos,
                                          data_strat['position'])  # sell signal
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['RSI'], 1).run()
            data_sized = self.regression_positions(data_pred, "RSI", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "RSI", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class MomVectorBacktester(StrategyCreator):
    def __init__(self,strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, momentum, reg_method, amount,
                 transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.momentum = momentum
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat

    def analyse_strategy(self, data):
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data["close"].plot(ax=ax1, color='g', lw=.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["close"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["close"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["Price", "Buy", "Sell"])
        # plt.title("Momentum Trading Strategy")
        # plt.show()

    def run_strategy(self, predictive_strat=False):
        ''' Backtests the trading strategy.
        '''
        data_strat = self.data.copy().dropna()
        data_strat['position'] = np.sign(data_strat['returns'].rolling(self.momentum).mean())
        if self.strat_type_pos==0:
            data_strat['position']=data_strat['position'].replace(-1,0)
        data_strat = data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['returns'], 1).run()
            data_sized = self.regression_positions(data_pred, "returns", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "returns", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class MRVectorBacktester(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, sma_window, threshold, reg_method,
                 amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.sma_window = sma_window
        self.threshold = threshold
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat

    def set_parameters(self, sma_window=None):
        if sma_window is not None:
            self.sma_window = self.data['close'].rolling(sma_window).mean()

    def analyse_strategy(self, data):
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data["close"].plot(ax=ax1, color='b', lw=.5)
        # data["sma"].plot(ax=ax1, color='g', lw=.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["close"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["close"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["Price", "SMA", "Buy", "Sell"])
        # plt.title("Mean Reversion Trading Strategy")
        # plt.show()

    def run_strategy(self, predictive_strat=False):
        self.set_parameters(int(self.sma_window))
        data_strat = self.data.copy().dropna()
        data_strat['sma']=self.sma_window
        data_strat['distance'] = data_strat['close'] - data_strat['sma']
        # sell signals
        data_strat['position'] = np.where(data_strat['distance'] > self.threshold,
                                          self.strat_type_pos, np.nan)
        # buy signals
        data_strat['position'] = np.where(data_strat['distance'] < -self.threshold,
                                          1, data_strat['position'])
        # crossing of current price and SMA (zero distance)
        data_strat['position'] = np.where(data_strat['distance'] *
                                          data_strat['distance'].shift(1) < 0,
                                          0, data_strat['position'])
        data_strat['position'] = data_strat['position'].ffill().fillna(0)
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['distance'], 1).run()
            data_sized = self.regression_positions(data_pred, "distance", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "distance", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class TurtleVectorBacktester(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, window_size, reg_method, amount,
                 transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.window_size = window_size
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat

    def analyse_strategy(self, data):
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data["close"].plot(ax=ax1, color='b', lw=.5)
        # data["high"].plot(ax=ax1, color='g', lw=2.)
        # data["low"].plot(ax=ax1, color='r', lw=2.)
        # data["avg"].plot(ax=ax1, color='y', lw=2.)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["close"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["close"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["Price", "Highs", "Lows", "Average", "Buy", "Sell"])
        # plt.title("Turtle Trading Strategy")
        # plt.show()

    def calculate_turtle(self):
        data = self.data.copy()
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
        data["orders"] = 0
        for k in range(1, len(data)):
            if data['long_entry'][k] and data['position'][k - 1] == 0:
                data.orders.values[k] = 1
                data.position.values[k] = 1
            elif data['short_entry'][k] and data['position'][k - 1] == 0:
                data.orders.values[k] = self.strat_type_pos
                data.position.values[k] = self.strat_type_pos
            elif data['short_exit'][k] and data['position'][k - 1] > 0:
                data.orders.values[k] = self.strat_type_pos
                data.position.values[k] = 0
            elif data['long_exit'][k] and data['position'][k - 1] < 0:
                data.orders.values[k] = 1
                data.position.values[k] = 0
            else:
                data.orders.values[k] = 0
        return data

    def run_strategy(self, predictive_strat=False):
        data_strat = self.calculate_turtle()
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['long_exit', 'long_entry', 'short_exit', 'short_entry'], 1).run()
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


class ParabolicSARBacktester(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, SAR_step, SAR_max, reg_method,
                 amount,
                 transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.SAR_step = SAR_step
        self.SAR_max = SAR_max
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat

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
                    data['trend'][i] = self.strat_type_pos  # Switch to downtrend
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
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['position'].diff()
        data = data.dropna(axis=0)
        # data["close"].plot(ax=ax1, color='b', lw=.5)
        # data["SAR"].plot(ax=ax1, color='g', lw=.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["close"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["close"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["Price", "SAR", "Buy", "Sell"])
        # plt.title("Parabolic SAR Trading Strategy")
        # plt.show()

    def run_strategy(self, predictive_strat=False):
        ''' Backtests the trading strategy. '''
        data_strat = self.calculate_parabolic_sar()

        # Define trading signals
        data_strat['position'] = np.where(data_strat['trend'] == 1, 1,
                                          self.strat_type_pos)  # Long when trend is up, short when trend is down
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['trend'], 1).run()
            data_sized = self.regression_positions(data_pred, "trend", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "trend", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class MACDStrategy(StrategyCreator):
    def __init__(self,strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, short_window, long_window,
                 signal_window, reg_method, amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data["macd"].plot(ax=ax1, color='g', lw=.5)
        # data["signal"].plot(ax=ax1, color='r', lw=.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["macd"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["macd"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["MACD", "Signal", "Buy", "Sell"])
        # plt.title("MACD Trading Strategy")
        # plt.show()

    def run_strategy(self, predictive_strat=False):
        data_strat = self.data.copy()
        # MACD calculation
        data_strat['ema_short'] = data_strat['close'].ewm(span=self.short_window, adjust=False).mean()
        data_strat['ema_long'] = data_strat['close'].ewm(span=self.long_window, adjust=False).mean()
        data_strat['macd'] = data_strat['ema_short'] - data_strat['ema_long']
        data_strat['signal'] = data_strat['macd'].ewm(span=self.signal_window, adjust=False).mean()
        data_strat['position'] = np.where(data_strat['macd'] > data_strat['signal'], 1, self.strat_type_pos)
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['macd', 'signal'], 1).run()
            data_sized = self.regression_positions(data_pred, "macd", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "macd", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class IchimokuStrategy(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, conversion_line_period,
                 base_line_period,
                 leading_span_b_period, displacement, reg_method, amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.conversion_line_period = conversion_line_period
        self.base_line_period = base_line_period
        self.leading_span_b_period = leading_span_b_period
        self.displacement = displacement
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat

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
                                   data['low'].rolling(window=self.leading_span_b_period).min()) / 2).shift(
            self.displacement)

        # Lagging Span
        data['lagging_span'] = data['close'].shift(-self.displacement)

        return data

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data["close"].plot(ax=ax1, color='b', lw=.5)
        # data["leading_span_A"].plot(ax=ax1, color='r', lw=0.5)
        # data["leading_span_B"].plot(ax=ax1, color='g', lw=0.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["close"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["close"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["Price", "Span A", "Span B", "Buy", "Sell"])
        # plt.title("Ichimoku Trading Strategy")
        # plt.show()

    def run_strategy(self, predictive_strat=False):
        data_strat = self.calculate_ichimoku()

        # Trading signals based on Ichimoku
        # Example: Go long when the Conversion Line crosses above the Base Line
        # and the close is above the Cloud; go short when the opposite is true.
        data_strat['position'] = np.where((data_strat['conversion_line'] > data_strat['base_line']) &
                                          (data_strat['close'] > data_strat['leading_span_A']) &
                                          (data_strat['close'] > data_strat['leading_span_B']), 1, 0)
        data_strat['position'] = np.where((data_strat['conversion_line'] < data_strat['base_line']) &
                                          (data_strat['close'] < data_strat['leading_span_A']) &
                                          (data_strat['close'] < data_strat['leading_span_B']), self.strat_type_pos,
                                          data_strat['position'])
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat,
                                    ['close', 'conversion_line', 'base_line', 'leading_span_A', 'leading_span_B'],
                                    1).run()
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


class StochasticOscillatorStrategy(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, k_window, d_window, buy_threshold,
                 sell_threshold, reg_method, amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.k_window = k_window
        self.d_window = d_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat

    def calculate_stochastic_oscillator(self):
        data = self.data.copy()
        data['14-high'] = data['high'].rolling(window=self.k_window).max()
        data['14-low'] = data['low'].rolling(window=self.k_window).min()
        data['%K'] = 100 * ((data['close'] - data['14-low']) / (data['14-high'] - data['14-low']))
        data['%D'] = data['%K'].rolling(window=self.d_window).mean()
        return data

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data['close'].plot(ax=ax1, color='b', lw=.5)
        # data["%K"].plot(ax=ax1, color='b', lw=.5)
        # data['%D'].plot(ax=ax1, color='g', lw=.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["close"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["close"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["%K", "%D", "Buy", "Sell"])
        # plt.title("Stochastic Oscillator Trading Strategy")
        # plt.show()

    def run_strategy(self, predictive_strat=False):
        data_strat = self.calculate_stochastic_oscillator()

        # Trading signals based on Stochastic Oscillator
        data_strat['position'] = np.where(data_strat['%K'] < data_strat['%D'], 1, 0)  # Buy signal
        data_strat['position'] = np.where(data_strat['%K'] > data_strat['%D'], self.strat_type_pos,
                                          data_strat['position'])  # Sell signal
        data_strat['Diff_%K_%D'] = data_strat['%D'] - data_strat['%K']
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['%K', '%D'], 1).run()
            data_sized = self.regression_positions(data_pred, "Diff_%K_%D", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "Diff_%K_%D", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class ADXStrategy(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, adx_period, di_period, threshold,
                 reg_method, amount,
                 transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.adx_period = adx_period
        self.di_period = di_period
        self.threshold = threshold
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data["close"].plot(ax=ax1, color='b', lw=.5)
        # data['+DI'].plot(ax=ax1, color='g', lw=.5)
        # data['-DI'].plot(ax=ax1, color='r', lw=.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["close"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["close"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["Price", "+DI", "-DI", "Buy", "Sell"])
        # plt.title("ADX Trading Strategy")
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

    def run_strategy(self, predictive_strat=False):
        data_strat = self.calculate_adx()

        # Trading signals based on ADX, +DI, and -DI
        data_strat['position'] = np.where(
            (data_strat['+DI'] > data_strat['-DI']) & (data_strat['ADX'] > self.threshold), 1, 0)  # Buy signal
        data_strat['position'] = np.where(
            (data_strat['-DI'] > data_strat['+DI']) & (data_strat['ADX'] > self.threshold), self.strat_type_pos,
            data_strat['position'])  # Sell signal

        data_strat['Diff_DI'] = data_strat['+DI'] - data_strat['-DI']
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['-DI', '+DI', 'ADX'], 1).run()
            data_sized = self.regression_positions(data_pred, "Diff_DI", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "Diff_DI", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class VolumeStrategy(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, volume_threshold,
                 volume_window, reg_method, amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.volume_threshold = volume_threshold
        self.volume_window = volume_window
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data["volume"].plot(ax=ax1, color='g', lw=.5)
        # data["average_volume"].plot(ax=ax1, color='r', lw=.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["volume"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["volume"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["Volume", "Average Volume", "Buy", "Sell"])
        # plt.title("Volume Trading Strategy")
        # plt.show()

    def run_strategy(self, predictive_strat=False):
        data_strat = self.data.copy()

        # Calculate the average volume over a specified period
        data_strat['average_volume'] = data_strat['volume'].rolling(window=self.volume_window).mean()

        # Identify significant volume increases - a volume spike
        data_strat['volume_spike'] = data_strat['volume'] > (data_strat['average_volume'] * self.volume_threshold)

        # Define a simple trading logic: buy when there is a volume spike, sell otherwise
        data_strat['position'] = np.where(data_strat['volume_spike'], 1, self.strat_type_pos)
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['volume_spike'], 1).run()
            data_sized = self.regression_positions(data_pred, "volume_spike", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "volume_spike", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class WilliamsRBacktester(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, lookback_period, overbought,
                 oversold,
                 reg_method, amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.lookback_period = lookback_period
        self.overbought = overbought
        self.oversold = oversold
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data["close"].plot(ax=ax1, color='g', lw=.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["close"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["close"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["Price", "Buy", "Sell"])
        # plt.title("William R Trading Strategy")
        # plt.show()

    def run_strategy(self, predictive_strat=False):
        data_strat = self.data.copy()
        high = data_strat['high'].rolling(self.lookback_period).max()
        low = data_strat['low'].rolling(self.lookback_period).min()
        data_strat['%R'] = -100 * (high - data_strat['close']) / (high - low)

        data_strat['position'] = np.where(data_strat['%R'] < -self.oversold, 1, 0)
        data_strat['position'] = np.where(data_strat['%R'] > -self.overbought, self.strat_type_pos, data_strat['position'])
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['%R'], 1).run()
            data_sized = self.regression_positions(data_pred, "%R", "pred_position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
        else:
            data_sized = self.regression_positions(data_strat, "%R", "position", self.reg_method)
            self.data_signal = data_sized['regularized_position'][-1]
            self.analyse_strategy(data_sized)
            return self.calculate_performance(data_sized)

    def generate_signal(self):
        self.run_strategy(self.predictive_strat)
        return self.data_signal


class VolatilityBreakoutBacktester(StrategyCreator):
    def __init__(self, strat_type_pos, frequency, data, symbol, risk_free_rate, start_date, end_date, volatility_window,
                 breakout_factor,
                 reg_method, amount, transaction_costs, predictive_strat):
        super().__init__(strat_type_pos, frequency, symbol, risk_free_rate, start_date, end_date, amount, transaction_costs,
                         predictive_strat)
        self.data = data
        self.volatility_window = volatility_window
        self.breakout_factor = breakout_factor
        self.reg_method = reg_method
        self.predictive_strat = predictive_strat
        self.set_parameters(volatility_window, breakout_factor)

    def set_parameters(self, volatility_window=None, breakout_factor=None):
        if volatility_window is not None:
            self.volatility_window = volatility_window
        if breakout_factor is not None:
            self.breakout_factor = breakout_factor

    def analyse_strategy(self, data):
        ''' Visualization of the strategy trades. '''
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, ylabel=f'{self.symbol} price in $')
        data['orders'] = data['regularized_position'].diff()
        data = data.dropna(axis=0)
        # data["close"].plot(ax=ax1, color='b', lw=.5)
        # data["lower_band"].plot(ax=ax1, color='r', lw=.5)
        # data["upper_band"].plot(ax=ax1, color='g', lw=.5)
        # ax1.plot(data.loc[data.orders >= 1.0].index,
        #          data["close"][data.orders >= 1.0],
        #          '^', markersize=7, color='k')
        # ax1.plot(data.loc[data.orders <= -1.0].index,
        #          data["close"][data.orders <= -1.0],
        #          'v', markersize=7, color='k')
        # plt.legend(["Price", "Lower Band", "Upper Band", "Buy", "Sell"])
        # plt.title("Volatility Breakout Trading Strategy")
        # plt.show()

    def run_strategy(self, predictive_strat=False):
        ''' Backtests the trading strategy. '''
        data_strat = self.data.copy().dropna()

        # Calculate ATR and bands
        data_strat['ATR'] = data_strat['close'].rolling(self.volatility_window).std()
        data_strat['upper_band'] = data_strat['close'] + data_strat['ATR'] * self.breakout_factor
        data_strat['lower_band'] = data_strat['close'] - data_strat['ATR'] * self.breakout_factor

        # Define trading signals
        data_strat['position'] = np.where(data_strat['close'] > data_strat['upper_band'], 1, np.nan)
        data_strat['position'] = np.where(data_strat['close'] < data_strat['lower_band'], self.strat_type_pos, data_strat['position'])
        data_strat['position'] = data_strat['position'].ffill().fillna(0)
        data_strat=data_strat.dropna(axis=0)
        if predictive_strat:
            data_pred = MLPredictor(data_strat, ['close', 'upper_band', 'lower_band'], 1).run()
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
