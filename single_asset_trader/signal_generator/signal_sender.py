import threading
from datetime import datetime, time
import pandas as pd
import pytz
from broker_interaction.broker_order import AlpacaTradingBot
from broker_interaction.broker_metrics import AlpacaPlatform
import time as counter
from indicators.performances_indicators import Returns, LogReturns, CumulativeReturns, \
    CumulativeLogReturns
from single_asset_trader.positions_pnl_tracker.manual_tracker import LiveStrategyTracker
from single_asset_trader.positions_pnl_tracker.prod_tracker_dashboard import ProductManagementApp


class LiveStrategyRunner:
    def __init__(self, strategy_name, strategy_class, strat_type_pos, optimization_results, frequency,
                 symbol, risk_free_rate, start_date, end_date, amount, transaction_costs, predictive_strat,
                 contract_multiplier, data_provider, broker_name, broker_config, dash_port):
        self.strategy_name = strategy_name
        self.strategy_class = strategy_class
        self.strat_type_pos=strat_type_pos
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
        self.broker_name = broker_name
        self.broker_config = broker_config
        self.current_positions = {name: 0 for name in optimization_results}
        self.broker_symbol= self.symbol.replace("-", "/") if '-' in self.symbol else self.symbol
        self.contract_multiplier = contract_multiplier
        self.real_time_data = None
        self.threads=[]
        self.dashboard_open=False
        if self.broker_name == 'alpaca':
            self.broker = AlpacaTradingBot(broker_config)
            self.trading_platform=AlpacaPlatform(self.broker_config, self.frequency)
        self.dash_port=dash_port

    def update_returns_data(self, downloaded_data):
        try:
            self.real_time_data=downloaded_data.copy()
            self.real_time_data['returns']=Returns().get_metric(self.real_time_data['close'])
            self.real_time_data['log_returns']=LogReturns().get_metric(self.real_time_data['returns'])
            self.real_time_data['creturns']=CumulativeReturns().get_metric(1, self.real_time_data['returns'])
            self.real_time_data['log_creturns']=CumulativeLogReturns().get_metric(1, self.real_time_data['log_returns'])
            self.logger_monitor(f"\n Data Available until {self.real_time_data.index[-1]}")
        except Exception as e:
            self.logger_monitor(f"Error in data fetching: {e}")

    def apply_strategy(self, strategy_name, strategy_class):
        try:
            if self.real_time_data is None:
                pass
            opti_results_strategy = self.optimization_results[strategy_name]['params']
            self.signal = strategy_class(self.strat_type_pos, self.frequency, self.real_time_data, self.symbol,
                                         self.risk_free_rate, self.start_date, self.end_date,
                                         amount=self.amount, transaction_costs=self.transaction_costs,
                                         predictive_strat=self.predictive_strat,
                                         **opti_results_strategy).generate_signal()
            if self.strat_type_pos==-1:
                #short selling only accept non-fractional order
                self.signal=round(self.signal)
            return self.execute_trade(strategy_name, self.signal)
            # Removed break; now it will loop continuously
        except Exception as e:
            if e.args[0] in ['qty', 'qty must be > 0']:
                pass
            else:
                self.logger_monitor(f"Error in strategy application: {e}")

    def execute_trade(self, strategy_name, signal):
        try:
            broker_position = self.trading_platform.get_symbol_position(self.broker_symbol)
        except Exception as e:
            broker_position=pd.DataFrame()
        current_position = 0
        if broker_position.empty:
            return self.place_order(strategy_name, signal, current_position)
        else:
            current_position = float(broker_position['qty'])
            if signal*self.contract_multiplier != current_position:
                return self.place_order(strategy_name, signal, current_position)


    def place_order(self, strategy_name, signal, current_position):
        new_position = (float(signal * self.contract_multiplier))
        self.logger_monitor(f'\nCurrent {self.symbol} Position: {current_position}\nRequested {self.symbol}  Position: {new_position}', False)
        # Calculate the percentage change between the new and current positions
        try:
            pct_change = (new_position - current_position) / (new_position)
        except ZeroDivisionError:
            pct_change=1
        # Initialize a dictionary to store trade information
        trade_info = {'symbol': self.symbol,'position': None,'order': None}
        # Determine if the change in position is significant enough to warrant a trade
        if abs(pct_change) > 0.5 and new_position!=current_position:
            order_qty = new_position - current_position
            side = 'buy' if order_qty > 0 else 'sell'
            self.broker.submit_order(self.broker_symbol, current_position, order_qty, side)
            trade_info.update({'position': new_position, 'order': order_qty})
            self.report_trade(self.symbol, strategy_name, side, abs(order_qty))
        else:
            order_abs_qty=(self.trading_platform.get_symbol_orders(self.broker_symbol)).at[0, 'qty']
            if (self.trading_platform.get_symbol_orders(self.broker_symbol)).at[0, 'side']=='buy':
                order_qty=float(order_abs_qty)
            else:
                order_qty=-float(order_abs_qty)
            trade_info.update({'position': current_position, 'order': order_qty})
            self.logger_monitor(f'Change In {self.symbol} Position Not Sufficient: No Trade Executed')
        df_trade_info = pd.DataFrame([trade_info])
        return df_trade_info


    def logger_monitor(self, message, *args, **kwargs):
        print(message)

    def report_trade(self, symbol, strategy_name, order_type, qty):
        self.logger_monitor(f'Trade Executed at {datetime.now().replace(second=0, microsecond=0)} on '
                            f'{symbol} following the {strategy_name} strategy: {order_type} {qty} units')

    def stop_loss(self):
        portfolio_history = self.trading_platform.\
                            get_broker_portfolio_history().iloc[-1]
        #0.1% of initial portfolio value
        if portfolio_history['base_value']-portfolio_history['equity']>0.001*portfolio_history['base_value']:
            return True

    def tracker(self, new_pos):
        #manual recap
        livestrat = LiveStrategyTracker(self.data_provider, self.symbol, self.frequency,
                                        self.start_date, self.end_date, self.amount)
        livestrat.get_asset_metrics(self.frequency, self.risk_free_rate,self.real_time_data['close'], new_pos)
        if not self.dashboard_open:
            product_management_port = self.dash_port + 1000
            product_manager_app = ProductManagementApp(self.trading_platform, self.symbol, self.frequency, product_management_port)
            prod_manager_run_server = threading.Thread(
                target=lambda: product_manager_app.run_server())
            prod_manager_run_server.daemon = True  # Ensures the thread is killed when the main program exits
            prod_manager_run_server.start()
            self.dashboard_open = True

    def run_once(self, downloaded_data):
        self.update_returns_data(downloaded_data)
        new_pos = self.apply_strategy(self.strategy_name, self.strategy_class)
        self.tracker(new_pos)
