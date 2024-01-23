import time
import datetime
import pytz
from broker_interaction.broker_order import AlpacaTradingBot
from broker_interaction.broker_metrics import AlpacaPlatform
from data_loader.data_retriever import DataRetriever
import numpy as np
import os

from indicators.performances_indicators import RiskFreeRate, Returns, LogReturns, CumulativeReturns, \
    CumulativeLogReturns


class LiveStrategyRunner:
    def __init__(self, strategy_name, strategy_class, optimization_results, frequency, symbol, risk_free_rate, start_date, end_date,
                 amount, transaction_costs, predictive_strat, contract_multiplier, data_provider, trading_platform, broker_config):
        self.strategy_name = strategy_name
        self.strategy_class = strategy_class
        self.optimization_results = optimization_results
        self.frequency = frequency
        self.symbol = symbol
        self.risk_free_rate=risk_free_rate
        self.start_date = start_date
        self.end_date = end_date
        self.amount = amount
        self.transaction_costs = transaction_costs
        self.predictive_strat=predictive_strat
        self.data_provider = data_provider
        self.trading_platform = trading_platform
        self.broker_config = broker_config
        self.current_positions = {name: 0 for name in optimization_results}
        self.contract_multiplier = contract_multiplier
        self.real_time_data = None
        if self.trading_platform == 'Alpaca':
            self.broker = AlpacaTradingBot(broker_config)

    def fetch_and_update_real_time_data(self):
        try:
            # Fetch and update data
            if self.data_provider == 'yfinance':
                self.real_time_data = DataRetriever(self.frequency, self.start_date, self.end_date) \
                    .yfinance_latest_data(self.symbol)[['open', 'high', 'low', 'close', 'volume']]
            else:
                self.real_time_data = \
                    DataRetriever(self.frequency, self.start_date, self.end_date).yfinance_latest_data(self.symbol)[
                        ['open', 'high', 'low', 'close', 'volume']]
            self.real_time_data['returns']=Returns().get_metric(self.real_time_data['close'])
            self.real_time_data['log_returns']=LogReturns().get_metric(self.real_time_data['returns'])
            self.real_time_data['creturns']=CumulativeReturns().get_metric(self.amount, self.real_time_data['returns'])
            self.real_time_data['log_creturns']=CumulativeLogReturns().get_metric(self.amount, self.real_time_data['log_returns'])
            self.logger_monitor(f"Data Available until {self.real_time_data.index[-1]}")
        except Exception as e:
            self.logger_monitor(f"Error in data fetching: {e}")

    def apply_strategy(self, strategy_name, strategy_class):
        try:
            if self.real_time_data is None:
                pass
            opti_results_strategy = self.optimization_results[strategy_name]['params']
            self.signal = strategy_class(self.frequency, self.real_time_data, self.symbol, self.risk_free_rate, self.start_date, self.end_date,
                                         amount=self.amount, transaction_costs=self.transaction_costs,
                                         predictive_strat=self.predictive_strat,
                                         **opti_results_strategy).generate_signal()

            if self.signal != 0:
                self.execute_trade(strategy_name, self.signal)
            # Removed break; now it will loop continuously
        except Exception as e:
            self.logger_monitor(f"Error in strategy application: {e}")
        if self.frequency['interval']=='1m':
            time.sleep(60)
        else:
            pass

    def execute_trade(self, strategy_name, signal):
        broker_positions = self.broker.get_positions()
        current_position = 0
        if broker_positions is None:
            self.place_order(strategy_name, signal, current_position)
        else:
            for i in range(len(broker_positions)):
                if broker_positions[i]._raw['symbol'] == self.symbol:
                    current_position = float(broker_positions[i]._raw['qty'])
                    if signal*self.contract_multiplier != current_position:
                        self.place_order(strategy_name, signal, current_position)

    def place_order(self, strategy_name, signal, current_position):
        self.logger_monitor(f'\nPosition: {current_position}\nRequested: {signal*self.contract_multiplier}', False)

        qty = (float(signal * self.contract_multiplier) - float(current_position))
        side = 'buy' if qty >0 else 'sell'

        # Alpaca order placement
        if '-' in self.symbol:
            self.symbol=self.symbol.replace("-","/")
        self.broker.submit_order(self.symbol, round(abs(qty)), side)

        self.report_trade(self.symbol, strategy_name, side, abs(qty))

    def logger_monitor(self, message, *args, **kwargs):
        print(message)

    def report_trade(self, symbol, strategy_name, order_type, qty):
        self.logger_monitor(f'Trade Executed at {self.real_time_data.index[-1]} on '
                            f'{symbol} following the {strategy_name} strategy: {order_type} {qty} units')

    def stop_loss(self):
        portfolio_history = AlpacaPlatform(self.broker_config).get_portfolio_history().iloc[-1]
        #0.1% of initial portfolio value
        if portfolio_history['base_value']-portfolio_history['equity']>0.001*portfolio_history['base_value']:
            return True

    def run(self):
        if self.frequency['interval']=='1d':
            self.fetch_and_update_real_time_data()
            self.apply_strategy(self.strategy_name, self.strategy_class)
        else:
            current_time = datetime.datetime.now(pytz.timezone('Europe/Paris')).time()
            stop_time = datetime.time(22, 0, 0)
            while current_time < stop_time:
                self.fetch_and_update_real_time_data()
                self.apply_strategy(self.strategy_name, self.strategy_class)
                if self.stop_loss():
                    print('Stop Loss Activated at {}'.format(self.real_time_data.index[-1]))
                    closed_positions=AlpacaPlatform(self.broker_config).close_open_positions()
                    for i in range(len(closed_positions)):
                        print(f'Positions closed at {self.real_time_data.index[-1]} on {closed_positions[i]["symbol"]}')
                    return

