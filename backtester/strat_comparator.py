# Compare and Run Strategies
import pandas as pd
from .strat_optimizer import StrategyOptimizer, RandomSearchAlgorithm, GridSearchAlgorithm, \
    SimulatedAnnealingAlgorithm, GeneticAlgorithm
from .strat_creator import StrategyCreator
from collections import defaultdict
import warnings
# To deactivate all warnings:
warnings.filterwarnings('ignore')

class StrategyRunner:
    def __init__(self, strategies, data_provider, symbol, start_date, end_date, param_grids, amount, transaction_costs,
                 iterations):
        """
        Initialize the StrategyRunner with given parameters.

        :param strategies: Dictionary of strategy names and their corresponding classes.
        :param symbol: The symbol for which the strategies will be run.
        :param data_provider: source of data
        :param start_date: The starting date for the strategies.
        :param end_date: The ending date for the strategies.
        :param param_grids: Parameter grids for optimizing the strategies.
        :param amount: Initial investment amount (default: 10000).
        :param transaction_costs: Costs per transaction (default: 0.0).
        """
        self.strategies = strategies
        self.data_provider=data_provider
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.amount = amount
        self.transaction_costs = transaction_costs
        self.param_grids = param_grids
        self.optimization_results = {}
        self.iterations=iterations

        # Load data once and reuse, improving efficiency
        self.data = StrategyCreator(self.symbol, self.start_date,
                                    self.end_date, self.amount, self.transaction_costs).get_data(self.data_provider)

    def _optimize_strategy(self, strategy_name, strategy_class, search_type):
        """
        Private method to optimize a single strategy.
        """
        optimizer = StrategyOptimizer(strategy_class, self.data, self.symbol, self.start_date, self.end_date,
                                      self.param_grids[strategy_name], self.amount, self.transaction_costs, search_type,
                                      self.iterations)
        return optimizer.optimize()


    def _update_optimization_results(self, strategy_name, best_params, best_performance, best_sharpe):
        """
        Private method to update the optimization results.
        """
        if strategy_name not in self.optimization_results or best_sharpe > self.optimization_results[strategy_name]['sharpe_ratio']:
            self.optimization_results[strategy_name] = {
                'params': best_params,
                'performance': best_performance,
                'sharpe_ratio': best_sharpe
            }

    def _append_strategy_results(self, comparison_data, strategy_name, strategy_tester):
        """
        Private method to append the results of a strategy to the comparison DataFrame.
        """
        if len(comparison_data['returns'])==0:
            comparison_data['returns']['creturns'] = strategy_tester.results['creturns']
        comparison_data['returns'][f'cstrategy_{strategy_name}'] = strategy_tester.results['cstrategy']
        comparison_data['positions'][f'positions_{strategy_name}']=strategy_tester.results['position']
        return comparison_data

    def optimize_strategies(self, search_type):
        """
        Optimize all strategies using a specified search type.

        :param search_type: The type of search to use for optimization ('random' or 'grid').
        :return: A dictionary of optimization results for each strategy.
        """
        for strategy_name, strategy_class in self.strategies.items():
            best_params, best_performance, best_sharpe = self._optimize_strategy(strategy_name, strategy_class, search_type)
            self._update_optimization_results(strategy_name, best_params, best_performance, best_sharpe)
        return self.optimization_results


    def test_all_search_types(self):
        """
        Test all search types and update the optimization results.

        :return: The best optimization results across all search types.
        """
        all_search_types = [RandomSearchAlgorithm(),GridSearchAlgorithm(),
                            SimulatedAnnealingAlgorithm(), GeneticAlgorithm()]
        for search_type in all_search_types:
            print(f"Testing with search type: {search_type}")
            self.optimize_strategies(search_type)
        return self.optimization_results



    def run_and_compare_strategies(self):
        """
        Run and compare the strategies based on the optimization results.

        :return: A tuple containing a summary of the best strategy results and a DataFrame for comparison.
        """
        comparison_data = defaultdict(pd.DataFrame)
        best_strats = {}


        for strategy_name, optimization_result in self.optimization_results.items():
            strategy_params = optimization_result['params']
            strategy_tester = self.strategies[strategy_name](self.data, self.symbol, self.start_date, self.end_date,
                                                             amount=self.amount, transaction_costs=self.transaction_costs,
                                                             **strategy_params)
            aperf, operf, sharpe = strategy_tester.run_strategy()
            best_strats[strategy_name] = {'params':optimization_result['params'], 'results':
                {'aperf': aperf, 'operf': operf, 'sharpe_ratio': sharpe}}
            comparison_data = self._append_strategy_results(comparison_data, strategy_name, strategy_tester)

        return best_strats, comparison_data

