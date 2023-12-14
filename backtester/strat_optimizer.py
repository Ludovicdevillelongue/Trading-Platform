import itertools
import math

import numpy as np
import random
import warnings
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')


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
        best_params={key:round(value,2) for key,value in best_params.items()}
        return best_params, best_performance, best_sharpe_ratio


class RandomSearchAlgorithm(OptimizationAlgorithm):
    def optimize(self, optimizer):
        return self.find_best_params(optimizer, self.random_evaluation)

    def random_evaluation(self, optimizer):
        for _ in range(optimizer.iterations):
            params=optimizer.generate_adaptive_params()
            # print(params)
            yield params


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
            # print(combination)
            yield dict(zip(optimizer.param_grids.keys(), combination))


class SimulatedAnnealingAlgorithm(OptimizationAlgorithm):
    def optimize(self, optimizer):
        return self.find_best_params(optimizer, self.simulated_annealing_evaluation)

    def simulated_annealing_evaluation(self, optimizer):
        current_params = optimizer.generate_adaptive_params()
        current_score = optimizer.test_strategy(current_params)[-1]
        temp = 1.0
        cooling_rate = 0.9

        for _ in range(optimizer.iterations):
            try:
                new_params = optimizer.generate_adaptive_params()
                new_score = optimizer.test_strategy(new_params)[-1]

                exponent = (current_score - new_score) / temp
                exponent = max(exponent, -700)  # Clamp to prevent overflow
                if new_score > current_score or math.exp(exponent) > random.random():
                    current_score = new_score
                    current_params = new_params
                    # print(current_params)
                    yield current_params

                temp *= cooling_rate
            except Exception as e:
                print(current_params)


class GeneticAlgorithm(OptimizationAlgorithm):
    def optimize(self, optimizer):
        return self.find_best_params(optimizer, self.genetic_algorithm_evaluation)

    def genetic_algorithm_evaluation(self, optimizer):
        population_size = optimizer.iterations
        mutation_rate = 0.1
        population = [optimizer.generate_adaptive_params() for _ in range(population_size)]

        scores = np.array([optimizer.test_strategy(params)[-1] for params in population])
        try:
            new_population = self.evolve_population(population, scores, mutation_rate)
            population = new_population
            # print(population)
            best_index = np.argmax(scores)
            yield population[best_index]
        except Exception as e:
            print(population)


    def evolve_population(self, population, scores, mutation_rate):
        new_population = []
        for _ in range(len(population) // 2):
            parent1, parent2 = self.select_parents(population, scores)
            try:
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1, mutation_rate), self.mutate(child2, mutation_rate)])
            except Exception as e:
                new_population.extend([parent1, parent2])
        return new_population

    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def select_parents(self, population, scores):
        probabilities = self.softmax(scores)
        parent_indices = np.random.choice(len(population), size=2, p=probabilities, replace=False)
        return population[parent_indices[0]], population[parent_indices[1]]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = {**{k: parent1[k] for k in list(parent1)[:crossover_point]}, **{k: parent2[k] for k in list(parent2)[crossover_point:]}}
        child2 = {**{k: parent2[k] for k in list(parent2)[:crossover_point]}, **{k: parent1[k] for k in list(parent1)[crossover_point:]}}
        return child1, child2

    def mutate(self, individual, mutation_rate):
        for key in individual:
            if random.random() < mutation_rate:
                if isinstance(individual[key], int):
                    individual[key] += random.randint(-1, 1)
                elif isinstance(individual[key], float):
                    individual[key] += random.uniform(-0.1, 0.1)
        return individual

class StrategyOptimizer:
    def __init__(self, strategy_class, data, symbol, start_date, end_date, param_grids, amount, transaction_costs, optimization_algorithm, iterations):
        self.strategy_class = strategy_class
        self.data = data
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.param_grids = param_grids
        self.amount = amount
        self.optimization_algorithm = optimization_algorithm
        self.transaction_costs = transaction_costs
        self.iterations = iterations
        self.param_history = []  # Track history of parameters

    def optimize(self):
        return self.optimization_algorithm.optimize(self)

    def generate_adaptive_params(self):
        # Adaptively adjust the search space based on past performance
        adaptive_param_grids = self.adapt_search_space()
        params = {}
        for key, param_range in adaptive_param_grids.items():
            if isinstance(param_range, tuple):
                min_val, max_val = param_range
                if isinstance(min_val, int):
                    params[key] = random.randint(min_val, max_val)
                else:
                    params[key] = random.uniform(min_val, max_val)
            elif isinstance(param_range, list):
                params[key] = random.choice(param_range)
        return params

    def adapt_search_space(self):
        if not self.param_history:
            return self.param_grids  # Return the original param_grids if no history

        # Analyze the parameter history to find the best performing parameters
        sorted_history = sorted(self.param_history, key=lambda x: x[1], reverse=True)  # Sort by performance (e.g., Sharpe ratio)
        top_performers = sorted_history[:max(1, len(sorted_history) // 5)]  # Take top 20% of performers

        # Adjust the search space based on top performers
        new_param_grids = {key: self._adjust_param_range(key, [params[0][key] for params in top_performers]) for key in self.param_grids}
        return new_param_grids

    def _adjust_param_range(self, param_key, top_values):
        # Adjust the range of a single parameter based on top performing values
        original_range = self.param_grids[param_key]
        if isinstance(original_range, tuple):
            min_val, max_val = original_range
            new_min = max(min_val, min(top_values) - (max_val - min_val) * 0.1)  # Contract range by 10%
            new_max = min(max_val, max(top_values) + (max_val - min_val) * 0.1)
            if isinstance(min_val, int):
                new_min, new_max = int(new_min), int(new_max)
            return new_min, new_max
        elif isinstance(original_range, list):
            # For discrete values, simply return the list of top values
            return list(set(top_values))


    def test_strategy(self, strategy_params):
        # Instantiate the strategy with provided parameters
        strategy_tester = self.strategy_class(self.data, self.symbol, self.start_date,
                                              self.end_date, amount=self.amount,
                                              transaction_costs=self.transaction_costs,
                                              **strategy_params)
        # Run the strategy
        aperf, operf, sharpe_ratio = strategy_tester.run_strategy()
        # Record the parameter performance
        self.param_history.append((strategy_params, sharpe_ratio))
        return aperf, operf, sharpe_ratio

