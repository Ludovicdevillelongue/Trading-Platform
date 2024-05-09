import os
import alpaca_trade_api as tradeapi
import yaml
import time as counter
class GetBrokersConfig:

    @staticmethod
    def key_secret_tc_url():
        config_file = os.path.join(os.path.dirname(__file__), '../config/broker_config.yml')
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

class AlpacaTradingBot:
    def __init__(self, config):
        self.config=config
        api_key = config['alpaca']['api_key']
        api_secret = config['alpaca']['api_secret']
        base_url = "https://paper-api.alpaca.markets"

        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    def submit_order(self, symbol, current_position, order_qty, side, order_type='market'):
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=abs(order_qty),
                side=side,
                type=order_type,
                time_in_force=self.config['alpaca']['time_in_force'][symbol]
            )
            print(f"Order submitted: {side} {abs(order_qty)} shares of {symbol}")

        except Exception as e:
            if abs(order_qty)==0:
                pass
            else:
                outstanding_qty = abs(order_qty)-abs(current_position)
                self.api.submit_order(
                    symbol=symbol,
                    qty=abs(current_position),
                    side=side,
                    type=order_type,
                    time_in_force=self.config['alpaca']['time_in_force'][symbol]
                )
                print(f"Order submitted: {side} {abs(current_position)} share of {symbol}")
                counter.sleep(2)
                self.api.submit_order(
                    symbol=symbol,
                    qty=abs(outstanding_qty),
                    side=side,
                    type=order_type,
                    time_in_force=self.config['alpaca']['time_in_force'][symbol]
                )
                print(f"Order submitted: {side} {abs(outstanding_qty)} share of {symbol}")


    def close_symbol_position(self, symbol):
        closed_positions=self.api.close_position(symbol)
        return closed_positions