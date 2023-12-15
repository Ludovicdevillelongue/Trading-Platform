import alpaca_trade_api as tradeapi
from datetime import datetime

import pandas as pd


class TradingPlatform:
    """Base class for all trading platforms."""

    def api_connection(self):
        raise NotImplementedError

    def get_account_info(self):
        raise NotImplementedError

    def get_portfolio_history(self):
        raise NotImplementedError

    def get_portfolio_metrics(self):
        raise NotImplementedError

class AlpacaPlatform(TradingPlatform):
    """Alpaca trading platform implementation."""
    def __init__(self, config):
        self.config=config
        self.get_api_connection()

    def get_api_connection(self):
        api_key = self.config['alpaca']['api_key']
        api_secret = self.config['alpaca']['api_secret']
        base_url = "https://paper-api.alpaca.markets"
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    def get_account_info(self):
        account = self.api.get_account()
        return {
            'currency':account.currency,
            'pending_transfer_in':account.pending_transfer_in,
            'created_at':account.created_at,
            'position_market_value':account.position_market_value,
            'cash': account.cash,
            'accrued_fees':account.accrued_fees,
            'buying_power': account.buying_power,
            'portfolio_value': account.portfolio_value
        }

    def get_orders(self):
        order=self.api.list_orders()
        return order


    def get_positions(self):
        positions=self.api.list_positions()
        return positions

    def get_assets(self):
        assets=self.api.list_assets()
        return assets

    def get_portfolio_history(self):
        portfolio_history=self.api.get_daily_portfolio_history()
        portfolio_history=pd.DataFrame(portfolio_history)
        portfolio_history['timestamp'] = portfolio_history['timestamp'].\
            apply(lambda x: pd.Timestamp.fromtimestamp(x))
        return portfolio_history


    def close_open_positions(self):
        closed_positions=self.api.close_open_positions()
        return closed_positions
