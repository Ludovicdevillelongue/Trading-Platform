# Compare and Run Strategies
import pandas as pd

from indicators.performances_indicators import RiskFreeRate
from .indicator_strat_creator import StrategyCreator
from collections import defaultdict
import warnings
# To deactivate all warnings:
from .strat_optimizer import StrategyOptimizer

warnings.filterwarnings('ignore')

class StrategyRunner:
    def __init__(self, strategies, data_provider, frequency, symbol, risk_free_rate, start_date, end_date, param_grids, opti_algo,
                 amount, transaction_costs, iterations, predictive_strat, strat_tester_csv):
        """
        Initialize the StrategyRunner with given parameters.

        :param strategies: Dictionary of strategy names and their corresponding classes.
        :param symbol: The symbol for which the trading_strategies will be run.
        :param risk_free_rate: value of risk free rate
        :param data_provider: source of data
        :param frequency: The frequency of the data.
        :param start_date: The starting date for the trading_strategies.
        :param end_date: The ending date for the trading_strategies.
        :param param_grids: Parameter grids for optimizing the trading_strategies.
        :param opti_algo: Optimization algorithm for optimizing the trading_strategies.
        :param amount: Initial investment amount (default: 10000).
        :param transaction_costs: Costs per transaction (default: 0.0).
        :param iterations: Number of iterations for optimizing the trading_strategies.
        :param predictive_strat: Whether to use predictive trading_strategies.
        """
        self.strategies = strategies
        self.data_provider=data_provider
        self.frequency = frequency
        self.symbol = symbol
        self.risk_free_rate=risk_free_rate
        self.start_date = start_date
        self.end_date = end_date
        self.amount = amount
        self.transaction_costs = transaction_costs
        self.param_grids = param_grids
        self.opti_algo = opti_algo
        self.optimization_results = {}
        self.iterations=iterations
        self.predictive_strat=predictive_strat
        self.strat_tester_csv=strat_tester_csv

        # Load data once and reuse, improving efficiency
        strat_creator=StrategyCreator(self.frequency, self.symbol, self.risk_free_rate, self.start_date,
                                    self.end_date, self.amount, self.transaction_costs,
                                    self.predictive_strat)
        self.data = strat_creator.get_data(self.data_provider)


    def _optimize_strategy(self, strategy_name, strategy_class, search_type):
        """
        Private method to optimize a single strategy.
        """
        optimizer = StrategyOptimizer(strategy_class, self.frequency, self.data, self.symbol, self.risk_free_rate, self.start_date, self.end_date,
                                      self.param_grids[strategy_name], self.amount, self.transaction_costs, search_type,
                                      self.iterations, self.predictive_strat, self.strat_tester_csv)
        return optimizer.optimize()


    def _update_optimization_results(self, strategy_name, search_type, best_params, best_performance, best_sharpe):
        """
        Private method to update the optimization results.
        """
        if strategy_name not in self.optimization_results or best_sharpe > self.optimization_results[strategy_name]['sharpe_ratio']:
            self.optimization_results[strategy_name] = {
                'search_type':str(type(search_type)).split(".")[2].split("'")[0],
                'params': best_params,
                'performance': best_performance,
                'sharpe_ratio': best_sharpe
            }


    def optimize_strategies(self, search_type):
        """
        Optimize all trading_strategies using a specified search type.

        :param search_type: The type of search to use for optimization ('random' or 'grid').
        :return: A dictionary of optimization results for each strategy.
        """
        for strategy_name, strategy_class in self.strategies.items():
            try:
                best_params, best_performance, best_sharpe=self._optimize_strategy(strategy_name, strategy_class, search_type)
            except Exception as e:
                print(f"Error optimizing {strategy_name}: {e}")
            self._update_optimization_results(strategy_name, search_type, best_params, best_performance, best_sharpe)
        return self.optimization_results


    def test_all_search_types(self):
        """
        Test all search types and update the optimization results.

        :return: The best optimization results across all search types.
        """
        all_search_types = self.opti_algo
        for search_type in all_search_types:
            print(f"Testing with search type: {search_type}")
            self.optimize_strategies(search_type)
        return self.optimization_results



    def _append_strategy_results(self, comparison_data, strategy_name, strategy_tester):
        """
        Private method to append the results of a strategy to the comparison DataFrame.
        """
        if len(comparison_data['returns'])==0:
            comparison_data['returns'][f'{self.symbol}_returns'] = strategy_tester.results['returns']
        if len(comparison_data['creturns'])==0:
            comparison_data['creturns']['cstrategy_HODL'] = strategy_tester.results['creturns']
        comparison_data['creturns'][f'cstrategy_{strategy_name}'] = strategy_tester.results['cstrategy']
        comparison_data['positions'][f'positions_{strategy_name}']=strategy_tester.results['regularized_position']
        return dict(comparison_data)


    def run_and_compare_strategies(self):
        """
        Run and compare the trading_strategies based on the optimization results.

        :return: A tuple containing a summary of the best strategy results and a DataFrame for comparison.
        """
        comparison_data = defaultdict(pd.DataFrame)
        best_strats = {}


        for strategy_name, optimization_result in self.optimization_results.items():
            strategy_params = optimization_result['params']
            strategy_tester = self.strategies[strategy_name](self.frequency, self.data,
                                                             self.symbol, self.risk_free_rate, self.start_date, self.end_date,
                                                             amount=self.amount, transaction_costs=self.transaction_costs,
                                                             predictive_strat=self.predictive_strat,
                                                             **strategy_params)
            aperf, operf, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown, \
            alpha, beta = strategy_tester.run_strategy()
            best_strats[strategy_name] = {'search_type':optimization_result['search_type'],
                                          'params':optimization_result['params'], 'results':
                {'aperf': aperf, 'operf': operf, 'sharpe_ratio': sharpe_ratio, 'sortino_ratio': sortino_ratio,
                 'calmar_ratio': calmar_ratio, 'max_drawdown': max_drawdown, 'alpha': alpha, 'beta': beta}}
            comparison_data = self._append_strategy_results(comparison_data, strategy_name, strategy_tester)

        return best_strats, dict(comparison_data)

