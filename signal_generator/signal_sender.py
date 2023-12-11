import threading
import yaml
from data_loader.data_retriever import DataRetriever
import numpy as np
import alpaca_trade_api as tradeapi
import os

class AlpacaTradingBot:
    def __init__(self, config_file):

        with open(config_file, 'r') as file:
            config = yaml.safe_load(config_file)

        api_key = config['alpaca']['api_key']
        api_secret = config['alpaca']['api_secret']
        base_url = "https://paper-api.alpaca.markets"

        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    def submit_order(self, symbol, qty, side, order_type='market', time_in_force='gtc'):
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            print(f"Order submitted: {side} {qty} shares of {symbol}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_positions(self):
        try:
            positions = self.api.list_positions()
            if not positions:
                print("No open positions.")
                return
            for position in positions:
                print(f"{position.qty} shares of {position.symbol} at ${position.avg_entry_price}")
        except Exception as e:
            print(f"An error occurred: {e}")

class LiveStrategyRunner:
    def __init__(self, strategy_name, strategy_class, optimization_results, symbol, start_date, end_date, amount, transaction_costs, data_provider, trading_platform):
        self.strategy_name = strategy_name
        self.strategy_class= strategy_class
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

        qty = (1 - 2 * current_position) * self.units
        side = 'buy' if signal == 1 else 'sell'
        self.report_trade(self.symbol, strategy_name, side, qty)
        self.current_positions[strategy_name] = signal
        qty=abs(signal)

        #Alpaca order placement
        current_script_path = os.path.dirname(__file__)
        AlpacaTradingBot(config_file=os.path.join(current_script_path, '..', 'broker_config.yml')).submit_order(self.symbol, qty, side)


    def report_trade(self, symbol, strategy_name, order_type, units):
        self.logger_monitor(f'Trade Executed at {self.real_time_data.index[-1]} on '
                            f'{symbol} following the {strategy_name} strategy: {order_type} {abs(units)} units')

    def fetch_and_update_real_time_data(self):

        self.real_time_data = DataRetriever(self.start_date, self.end_date).yfinance_latest_data(self.symbol)['price'].to_frame()
        self.real_time_data['return'] = np.log(self.real_time_data['price'] / self.real_time_data['price'].shift(1))
        self.logger_monitor(f"Data Available until {self.real_time_data.index[-1]}")
        # time.sleep(60)

    def apply_strategy(self, strategy_name, strategy_class):
        while self.running:
            if self.real_time_data is None:
                continue
            opti_results_strategy = self.optimization_results[strategy_name]['params']
            self.signal = strategy_class(self.real_time_data, self.symbol, self.start_date, self.end_date,
                                    amount=self.amount, transaction_costs=self.transaction_costs, **opti_results_strategy).generate_signal()

            if self.signal != 0:
                self.execute_trade(strategy_name, self.signal)
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
        self.fetch_and_update_real_time_data()
        self.apply_strategy(self.strategy_name, self.strategy_class)


    def stop(self):
        self.running = False
