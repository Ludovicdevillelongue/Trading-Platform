import threading
from datetime import datetime, time
import pandas as pd
import pytz
from broker_interaction.broker_order import AlpacaTradingBot
from broker_interaction.broker_metrics import AlpacaPlatform
from data_loader.data_retriever import DataManager
import time as counter
import os
from indicators.performances_indicators import Returns, LogReturns, CumulativeReturns, \
    CumulativeLogReturns
from positions_pnl_tracker.manual_tracker import LiveStrategyTracker
from positions_pnl_tracker.pnl_tracker_dashboard import PortfolioManagementApp


class LiveStrategyRunner:
    def __init__(self, strategy_name, strategy_class, optimization_results, frequency_yaml, frequency, symbol, risk_free_rate, start_date, end_date,
                 amount, transaction_costs, predictive_strat, contract_multiplier, data_provider, trading_platform, broker_config):
        self.strategy_name = strategy_name
        self.strategy_class = strategy_class
        self.optimization_results = optimization_results
        self.frequency_yaml=frequency_yaml
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
        self.threads=[]
        self.dashboard_open=False
        if self.trading_platform == 'Alpaca':
            self.broker = AlpacaTradingBot(broker_config)

    def fetch_and_update_real_time_data(self):
        try:
            # Fetch and update data
            if self.data_provider == 'yfinance':
                self.real_time_data = DataManager(self.frequency, self.start_date, self.end_date) \
                    .yfinance_download(self.symbol)[['open', 'high', 'low', 'close', 'volume']]
            else:
                pass
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
            self.signal = strategy_class(self.frequency, self.real_time_data, self.symbol,
                                         self.risk_free_rate, self.start_date, self.end_date,
                                         amount=self.amount, transaction_costs=self.transaction_costs,
                                         predictive_strat=self.predictive_strat,
                                         **opti_results_strategy).generate_signal()

            if self.signal != 0:
                return self.execute_trade(strategy_name, self.signal)
            # Removed break; now it will loop continuously
        except Exception as e:
            self.logger_monitor(f"Error in strategy application: {e}")

    def execute_trade(self, strategy_name, signal):
        broker_positions = AlpacaPlatform(self.broker_config).get_positions()
        current_position = 0
        broker_symbol=self.symbol.replace("-", "")
        if broker_positions.empty:
            return self.place_order(strategy_name, signal, current_position)
        else:
            for i in range(len(broker_positions)):
                if broker_positions.loc[i,'symbol'] == broker_symbol:
                    current_position = float(broker_positions.loc[i,'qty'])
                    if signal*self.contract_multiplier != current_position:
                        return self.place_order(strategy_name, signal, current_position)


    def place_order(self, strategy_name, signal, current_position):
        dict_pos={}
        new_position = (float(signal * self.contract_multiplier))
        self.logger_monitor(f'\nPosition: {current_position}\nRequested: {new_position}', False)

        #check if the requested position changed sufficiently compared to the old one -> regularize positions!!
        pct_change=(new_position-current_position)/new_position
        if abs(pct_change)>0.5:
            qty = ((float(new_position)) - float(current_position))
        else:
            qty = 0
        side = 'buy' if qty >0 else 'sell'

        # Alpaca order placement
        if '-' in self.symbol:
            broker_symbol=self.symbol.replace("-","/")
            self.broker.submit_order(broker_symbol, abs(qty), side)
        else:
            self.broker.submit_order(self.symbol, abs(qty), side)

        #save position
        dict_pos['symbol']=self.symbol
        dict_pos['position']=new_position
        dict_pos['order']=qty
        df_pos = pd.DataFrame.from_dict(dict_pos, orient='index').T
        self.report_trade(self.symbol, strategy_name, side, abs(qty))
        return df_pos

    def logger_monitor(self, message, *args, **kwargs):
        print(message)

    def report_trade(self, symbol, strategy_name, order_type, qty):
        self.logger_monitor(f'Trade Executed at {datetime.now().replace(second=0, microsecond=0)} on '
                            f'{symbol} following the {strategy_name} strategy: {order_type} {qty} units')

    def stop_loss(self):
        portfolio_history = AlpacaPlatform(self.broker_config).get_broker_portfolio_history().iloc[-1]
        #0.1% of initial portfolio value
        if portfolio_history['base_value']-portfolio_history['equity']>0.001*portfolio_history['base_value']:
            return True

    def tracker(self, df_pos):
        #manual recap
        try:
            LiveStrategyTracker(self.data_provider, self.symbol, self.frequency_yaml[self.data_provider]['minute'],
                                                     self.start_date, self.end_date, self.amount).\
                get_asset_metrics(self.frequency, self.risk_free_rate, df_pos)
        except Exception as e:
            pass
        alpaca_platform = AlpacaPlatform(self.broker_config)
        alpaca_platform.get_portfolio_metrics(self.frequency, self.symbol, self.risk_free_rate, self.real_time_data['returns'])
        #brokerage platform reca
        # if not self.dashboard_open:
        #     alpaca_platform = AlpacaPlatform(self.broker_config)
        #     alpaca_platform.get_portfolio_metrics(self.frequency, self.symbol, self.risk_free_rate, self.real_time_data['returns'])
        #     portfolio_manager_app = PortfolioManagementApp(alpaca_platform, self.frequency, self.symbol,
        #                                                    self.risk_free_rate, self.real_time_data['returns'])
        #     portfolio_server_thread = threading.Thread(target=portfolio_manager_app.run_server)
        #     portfolio_server_thread.start()
        #     self.threads.append(portfolio_server_thread)
        #
        #     # Start the PortfolioManagementApp browser thread
        #     portfolio_browser_thread = threading.Thread(target=portfolio_manager_app.open_browser)
        #     portfolio_browser_thread.start()
        #     self.threads.append(portfolio_browser_thread)
        #
        #     self.dashboard_open = True

    def tracker_thread(self, df_pos):
        """Function to run the tracker in a separate thread."""
        self.tracker(df_pos)


    def run(self):
        if self.frequency['interval']=='1d':
            self.fetch_and_update_real_time_data()
            df_pos=self.apply_strategy(self.strategy_name, self.strategy_class)
            tracker_tread=threading.Thread(target=self.tracker_thread, args=(df_pos,))
            tracker_tread.start()
            self.threads.append(tracker_tread)
        else:
            current_time = datetime.now(pytz.timezone('Europe/Paris')).time()
            stop_time = time(22, 0, 0)
            if self.symbol!="BTC-USD":
                while current_time < stop_time:
                    self.fetch_and_update_real_time_data()
                    df_pos=self.apply_strategy(self.strategy_name, self.strategy_class)
                    tracker_tread=threading.Thread(target=self.tracker_thread, args=(df_pos,)).start()
                    self.threads.append(tracker_tread)
                    if self.stop_loss():
                        print('Stop Loss Activated at {}'.format(self.real_time_data.index[-1]))
                        closed_positions = self.broker.close_open_positions()
                        for i in range(len(closed_positions)):
                            print(f'Positions closed at {self.real_time_data.index[-1]} on {closed_positions[i]["symbol"]}')
                    counter.sleep(60)
            else:
                while True:
                    self.fetch_and_update_real_time_data()
                    df_pos=self.apply_strategy(self.strategy_name, self.strategy_class)
                    self.tracker(df_pos)
                    # tracker_tread=threading.Thread(target=self.tracker_thread, args=(df_pos,)).start()
                    # self.threads.append(tracker_tread)
                    if self.stop_loss():
                        print('Stop Loss Activated at {}'.format(self.real_time_data.index[-1]))
                        closed_positions = self.broker.close_open_positions()
                        for i in range(len(closed_positions)):
                            print(f'Positions closed at {self.real_time_data.index[-1]} on {closed_positions[i]["symbol"]}')
                    counter.sleep(60)





