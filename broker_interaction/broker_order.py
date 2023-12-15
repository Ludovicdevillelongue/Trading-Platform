import os

import alpaca_trade_api as tradeapi
import yaml

class GetBrokersConfig:

    @staticmethod
    def key_secret_url():
        config_file = os.path.join(os.path.dirname(__file__), '../config/broker_config.yml')
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

class AlpacaTradingBot:
    def __init__(self, config):

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
            outstanding_qty=qty
            while outstanding_qty>0:
                self.api.submit_order(
                    symbol=symbol,
                    qty=1,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force
                )
                outstanding_qty-=1
                print(f"Order submitted: {side} {outstanding_qty} shares of {symbol}")

    def get_positions(self):
        try:
            positions = self.api.list_positions()
            if not positions:
                pass
            if len(positions)==0:
                pass
            else:
                return positions
        except Exception as e:
            print(f"An error occurred: {e}")
