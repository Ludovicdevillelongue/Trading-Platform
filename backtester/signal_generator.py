import threading
import time
from data_loader.data_retriever import DataRetriever
import numpy as np


class LiveStrategyRunner:
    def __init__(self, strategies, optimization_results, symbol, start_date, end_date, amount, transaction_costs, data_provider, trading_platform):
        self.strategies = strategies
        self.optimization_results = optimization_results
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.amount = amount
        self.transaction_costs = transaction_costs
        self.data_provider = data_provider
        self.trading_platform = trading_platform
        self.current_positions = {name: 0 for name in optimization_results}
        self.running = True
        self.units = 1
        self.real_time_data = None

    def place_order(self, strategy_name, signal):
        current_position = self.current_positions[strategy_name]
        self.logger_monitor(f'\nPosition: {current_position}\nSignal: {signal}', False)

        if current_position == signal:
            self.logger_monitor('*** NO TRADE PLACED ***')
            return

        units = (1 - 2 * current_position) * self.units
        order_type = 'buy' if signal == 1 else 'sell'
        self.report_trade(self.symbol, strategy_name, order_type, units)
        self.current_positions[strategy_name] = signal

    def report_trade(self, symbol, strategy_name, order_type, units):
        self.logger_monitor(f'Trade Executed at {self.real_time_data.index[-1]} on '
                            f'{symbol} following the {strategy_name} strategy: {order_type} {abs(units)} units')

    def fetch_and_update_real_time_data(self):
        while self.running:
            self.real_time_data = DataRetriever(self.start_date, self.end_date).yfinance_latest_data(self.symbol)['price'].to_frame()
            self.real_time_data['return'] = np.log(self.real_time_data['price'] / self.real_time_data['price'].shift(1))
            self.logger_monitor(f"Data Available until {self.real_time_data.index[-1]}")
            time.sleep(60)

    def apply_strategy(self, strategy_name, strategy_class):
        while self.running:
            if self.real_time_data is None:
                continue
            opti_results_strategy = self.optimization_results[strategy_name]['params']
            signal = strategy_class(self.real_time_data, self.symbol, self.start_date, self.end_date,
                                    amount=self.amount, transaction_costs=self.transaction_costs, **opti_results_strategy).generate_signal()

            if signal != 0:
                self.execute_trade(strategy_name, signal)
                break
            else:
                break

    def execute_trade(self, strategy_name, signal):
        current_position = self.current_positions[strategy_name]
        if signal != current_position:
            self.place_order(strategy_name, signal)

    def logger_monitor(self, message, *args, **kwargs):
        print(message)

    def run(self):
        data_thread = threading.Thread(target=self.fetch_and_update_real_time_data)
        data_thread.start()

        threads = []
        for strategy_name, strategy_class in self.strategies.items():
            strategy_thread = threading.Thread(target=self.apply_strategy, args=(strategy_name, strategy_class))
            threads.append(strategy_thread)
            strategy_thread.start()

        for thread in threads:
            thread.join()

        self.running = False
        data_thread.join()

    def stop(self):
        self.running = False
