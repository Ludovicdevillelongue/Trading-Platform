import itertools
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from abc import ABC, abstractmethod

class OptimizationAlgorithm(ABC):
    @abstractmethod
    def optimize(self, optimizer):
        pass

    def find_best_params(self, optimizer, evaluate):
        best_performance = float('-inf')
        best_params = None
        best_sharpe_ratio = float('-inf')

        for params in evaluate(optimizer):
            aperf, operf, sharpe_ratio = optimizer.test_strategy(params)

            if sharpe_ratio > best_sharpe_ratio:
                best_performance = aperf
                best_params = params
                best_sharpe_ratio = sharpe_ratio

        return best_params, best_performance, best_sharpe_ratio

class RandomSearchAlgorithm(OptimizationAlgorithm):
    def optimize(self, optimizer):
        return self.find_best_params(optimizer, self.random_evaluation)

    def random_evaluation(self, optimizer):
        for _ in range(optimizer.iterations):
            yield optimizer.generate_random_params()


class GridSearchAlgorithm(OptimizationAlgorithm):
    def optimize(self, optimizer):
        return self.find_best_params(optimizer, self.grid_evaluation)

    def grid_evaluation(self, optimizer):
        param_values = []
        for key, value in optimizer.param_grids.items():
            if isinstance(value, tuple):
                min_val, max_val = value
                if isinstance(min_val, int):
                    param_values.append(range(min_val, max_val + 1))
                else:
                    param_values.append(np.arange(min_val, max_val, 0.1).tolist())
            elif isinstance(value, list):
                param_values.append(value)

        all_combinations = itertools.product(*param_values)

        # Limit the number of combinations to optimizer.iterations
        limited_combinations = itertools.islice(all_combinations, optimizer.iterations)

        for combination in limited_combinations:
            yield dict(zip(optimizer.param_grids.keys(), combination))

class StrategyOptimizer:
    def __init__(self, strategy_class, data, symbol, start_date, end_date, param_grids, amount, transaction_costs, optimization_algorithm, iterations):
        self.strategy_class = strategy_class
        self.data = data
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.param_grids = param_grids
        self.amount = amount
        self.transaction_costs = transaction_costs
        self.optimization_algorithm = optimization_algorithm
        self.iterations = iterations

    def optimize(self):
        return self.optimization_algorithm.optimize(self)

    def generate_random_params(self):
        params = {}
        for key, param_range in self.param_grids.items():
            if isinstance(param_range, tuple):
                min_val, max_val = param_range
                if isinstance(min_val, int):
                    params[key] = random.randint(min_val, max_val)
                else:
                    params[key] = random.uniform(min_val, max_val)
            elif isinstance(param_range, list):
                params[key] = random.choice(param_range)
        return params


    def test_strategy(self, strategy_params):
        # Instantiate the strategy with provided parameters
        strategy_tester = self.strategy_class(self.data, self.symbol, self.start_date,
                                              self.end_date, amount=self.amount,
                                              transaction_costs=self.transaction_costs,
                                              **strategy_params)
        # Run the strategy
        aperf, operf, sharpe_ratio = strategy_tester.run_strategy()

        return aperf, operf, sharpe_ratio
