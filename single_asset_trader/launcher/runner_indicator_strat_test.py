import os
import logging
from datetime import timedelta, datetime
import threading

import pandas as pd
import pytz
import time as counter
from datetime import time
import yaml

from data_loader.data_retriever import DataManager
from indicators.performances_indicators import RiskFreeRate
from broker_interaction.broker_order import GetBrokersConfig
from single_asset_trader.trading_strategies.strat_optimizer import RandomSearchAlgorithm
from single_asset_trader.trading_strategies.indicator_strat_creator import SMAVectorBacktester
from single_asset_trader.trading_strategies.strat_comparator import StrategyRunner
from single_asset_trader.backtester_tracker.backtester_dashboard import BacktestApp
from single_asset_trader.signal_generator.signal_sender import LiveStrategyRunner


class MultiSymbolTrader:
    def __init__(self, symbols, invested_amount, contract_multiplier, iterations, predictive_strat, data_provider,
                 broker, strat_type):
        self.symbols = symbols
        self.invested_amount = invested_amount
        self.contract_multiplier = contract_multiplier
        self.iterations = iterations
        self.predictive_strat = predictive_strat
        self.data_provider = data_provider
        self.broker = broker
        self.strat_type=strat_type

        self.strategies = {
            'SMA': SMAVectorBacktester,
        }
        self.regression_methods = ['no_regression']
        self.param_grids = {
            'SMA': {'sma_short': (1, 10), 'sma_long': (10, 30), 'reg_method': self.regression_methods},
        }
        self.opti_algo = [RandomSearchAlgorithm()]
        self.broker_config = GetBrokersConfig.key_secret_tc_url()

        with open(r'../trading_platform/config/data_frequency.yml') as file:
            self.frequency_yaml = yaml.safe_load(file)
        self.frequency = self.frequency_yaml[self.data_provider]['minute']

        with open(r'../trading_platform/config/strat_type_pos.yml') as file:
            self.strat_type_pos_yaml = yaml.safe_load(file)
        self.strat_type_pos = float(self.strat_type_pos_yaml[self.strat_type])

        self.risk_free_rate = RiskFreeRate().get_metric()
        self.start_date = '2023-11-15 00:00:00'
        self.end_date = ((datetime.now(pytz.timezone('US/Eastern')) - timedelta(minutes=2)).replace(second=0)).strftime(
            "%Y-%m-%d %H:%M:%S")
        self.transaction_costs = self.broker_config[self.broker]['transaction_costs']

    def schedule_live_runner(self, strat_run, symbol, interval=60):
        def run_continuously():
            stop_time = time(22, 0, 0)  # 10 PM Paris time
            while True:
                current_time = datetime.now(pytz.timezone('Europe/Paris')).time()
                if current_time >= stop_time:
                    break
                data_download = self.download_data(self.symbols)
                symbol_data = data_download.xs(symbol, level=1, axis=1)
                strat_run.run_once(symbol_data)
                counter.sleep(interval)

        thread = threading.Thread(target=run_continuously)
        thread.start()
        return thread

    def run(self):
        threads = []
        data=self.download_data(self.symbols)
        base_port = 8100
        for i, symbol in enumerate(self.symbols):
            symbol_data=data.xs(symbol, level=1, axis=1)
            t = threading.Thread(target=self.run_for_symbol, args=(symbol,symbol_data,  base_port + i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    def download_data(self, symbols):
        if self.data_provider=='yfinance':
            data = DataManager(self.frequency).yfinance_download(symbols) \
                [['open', 'high', 'low', 'close', 'volume']]
        else:
            data=pd.DataFrame()
        return data

    def run_for_symbol(self, symbol, data, port):
        strat_tester_csv = os.path.join(os.path.dirname(__file__),
                                        f'../backtester_tracker/{symbol}_strat_tester_recap.csv')
        with open(strat_tester_csv, 'w') as file:
            pass

        runner = StrategyRunner(data, self.strategies, self.strat_type_pos, self.data_provider, self.frequency, symbol,
            self.risk_free_rate, self.start_date, self.end_date, self.param_grids, self.opti_algo,
            self.invested_amount, self.transaction_costs, self.iterations, self.predictive_strat,
            strat_tester_csv)

        print(f"Optimizing trading strategies for {symbol}...")
        start_time_opti = counter.time()
        optimization_results = runner.test_all_search_types()
        end_time_opti = counter.time()
        time_diff = end_time_opti - start_time_opti
        print(
            f'Elapsed time for optimization of {symbol}: {int(time_diff // 60)} minutes and {int(time_diff % 60)} seconds')
        print("Optimized results: %s", optimization_results)

        print(f"Running and comparing trading strategies for {symbol}...")
        best_strats, comparison_data = runner.run_and_compare_strategies()
        app = BacktestApp(best_strats, comparison_data, symbol, port)
        threading.Thread(target=app.run_server).start()
        threading.Thread(target=app.open_browser).start()
        best_strat = max(best_strats, key=lambda k: best_strats[k]['results']['sharpe_ratio'])
        strat_run = LiveStrategyRunner(best_strat, self.strategies[best_strat],
                                       self.strat_type_pos, optimization_results,
                                       self.frequency, symbol, self.risk_free_rate, self.start_date,
                                       self.end_date, self.invested_amount,
                                       self.transaction_costs, self.predictive_strat,
                                       self.contract_multiplier, self.data_provider,
                                       self.broker, self.broker_config, port)
        self.schedule_live_runner(strat_run, symbol)


if __name__ == '__main__':
    symbols = ['TSLA', 'MSFT']
    invested_amount = 100000
    contract_multiplier = 2
    iterations = 10
    predictive_strat = False
    data_provider = 'yfinance'
    broker = 'alpaca'
    strat_type='long_short'

    trader = MultiSymbolTrader(symbols, invested_amount, contract_multiplier, iterations, predictive_strat,
                               data_provider, broker, strat_type)
    trader.run()
