import alpaca_trade_api as tradeapi
import yaml

class AlpacaTradingBot:
    def __init__(self, config_file):

        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

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
            if positions.empty:
                print("No open positions.")
            else:
                return positions

        except Exception as e:
            print(f"An error occurred: {e}")
