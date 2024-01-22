import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

from backtester.indicator_strat_creator import StrategyCreator


class LRVectorBacktester(StrategyCreator):

    def __init__(self, frequency, data, symbol, start_date, end_date, lags, train_percent, model, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data=data
        self.lags=lags
        self.cols = [f'lag_{lag}' for lag in range(1, self.lags + 1)]
        self.train_percent=train_percent
        self.model= model
        self.reg_method=reg_method
        self.train_start = None
        self.train_end = None
        self.test_start = None
        self.test_end = None

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
        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        self.results=self.regression_positions(self.results, "returns", self.reg_method)
        self.analyse_strategy(self.results)
        return self.calculate_performance(self.results)



    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        latest_row=self.fit_model_for_signal(self.lags, self.model)
        if not latest_row.empty and self.reg is not None:
            prediction = np.dot(latest_row[self.cols], self.reg)
            return int(np.sign(prediction))
        else:
            return 0  # Return neutral signal if data is insufficient


class ScikitVectorBacktester(StrategyCreator):
    def __init__(self, frequency, data, symbol, start_date, end_date, lags, train_percent, model, reg_method, amount, transaction_costs):
        super().__init__(frequency, symbol, start_date, end_date, amount, transaction_costs)
        self.data=data
        self.lags=lags
        self.cols = [f'lag_{lag}' for lag in range(1, self.lags + 1)]
        self.train_percent=train_percent
        self.model=model
        self.reg_method=reg_method
        self.train_start = None
        self.train_end = None
        self.test_start = None
        self.test_end = None

        if model == 'linear':
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
        # Implement the rest of the strategy logic similar to SMAVectorBacktester
        self.results = self.regression_positions(self.results, "close", self.reg_method)
        self.results = self.results['regularized_position'][-1]
        self.analyse_strategy(self.results)
        return self.calculate_performance(self.results)



    def generate_signal(self):
        ''' Generates a trading signal for the most recent data point. '''
        latest_row=self.fit_model_for_signal(self.lags, self.model)
        if not latest_row.empty:
            prediction = self.model.predict(latest_row[self.cols])
            return int(np.sign(prediction))
        else:
            return 0



