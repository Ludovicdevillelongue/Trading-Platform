import alpaca_trade_api as tradeapi
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from data_loader.data_retriever import DataManager
from indicators.performances_indicators import SharpeRatio, SortinoRatio, MaxDrawdown, CalmarRatio, Beta, Alpha, \
    RiskFreeRate, Returns, CumulativeReturns


class TradingPlatform:
    """Base class for all trading platforms."""

    def api_connection(self):
        raise NotImplementedError

    def get_account_info(self):
        raise NotImplementedError

    def get_orders(self):
        raise NotImplementedError

    def get_positions(self):
        raise NotImplementedError

    def get_portfolio_history(self):
        raise NotImplementedError

    def get_portfolio_metrics(self, frequency, symbol, risk_free_rate, df_benchmark):
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
        dict_account_info={
            'currency':account.currency,
            'pending_transfer_in':account.pending_transfer_in,
            'created_at':account.created_at,
            'position_market_value':account.position_market_value,
            'cash': account.cash,
            'accrued_fees':account.accrued_fees,
            'buying_power': account.buying_power,
            'portfolio_value': account.portfolio_value
        }
        df_account_info=pd.DataFrame.from_dict(dict_account_info, orient='index').T
        return df_account_info

    def get_orders(self):
        orders=self.api.list_orders(status='all')
        orders_list=[order._raw for order in orders]
        df_orders=pd.DataFrame(orders_list)
        if df_orders.empty:
            return pd.DataFrame()
        else:
            return df_orders[['created_at', 'filled_at', 'asset_id', 'symbol',
                         'asset_class', 'qty', 'filled_qty', 'order_type', 'side', 'filled_avg_price',
                         'time_in_force', 'limit_price', 'stop_price']]


    def get_positions(self):
        positions=self.api.list_positions()
        positions_list=[position._raw for position in positions]
        df_positions=pd.DataFrame(positions_list)
        return df_positions

    def get_assets(self):
        assets=self.api.list_assets()
        return assets

    def get_portfolio_history(self):
        portfolio_history=self.api.get_daily_portfolio_history()
        portfolio_history=pd.DataFrame(portfolio_history)
        portfolio_history['timestamp'] = portfolio_history['timestamp'].\
            apply(lambda x: pd.Timestamp.fromtimestamp(x))
        portfolio_history.drop(index=portfolio_history.index[-1],axis=0,inplace=True)
        return portfolio_history



    def get_portfolio_metrics(self, frequency, symbol, risk_free_rate, df_benchmark):
        dict_key_metrics={}
        df_ptf=self.get_portfolio_history()
        df_ptf.set_index('timestamp', inplace=True)
        df_ptf=df_ptf.tz_localize('Europe/Paris')
        try:
            df_benchmark=df_benchmark.tz_localize('Europe/Paris')
        except Exception as e:
            pass
        # Calculate returns
        df_ptf['ptf_returns'] =Returns().get_metric(df_ptf['equity'])
        df_ptf['ptf_creturns'] = CumulativeReturns().get_metric(df_ptf['base_value'].iloc[0], df_ptf['ptf_returns'])
        df_ptf=df_ptf.dropna(axis=0)
        df_ptf_vs_bench = df_ptf.merge(df_benchmark, left_index=True, right_index=True, how='outer')
        df_ptf_vs_bench=df_ptf_vs_bench.dropna(axis=0)

        # Calculate Performance Indicators

        try:
            dict_key_metrics['sharpe_ratio']  = SharpeRatio(frequency, risk_free_rate).calculate(df_ptf['ptf_returns'])
        except Exception as e:
            dict_key_metrics['sharpe_ratio'] =0
        dict_key_metrics['sortino_ratio'] =SortinoRatio(frequency, risk_free_rate).calculate(df_ptf['ptf_returns'])

        dict_key_metrics['max_drawdown'] =MaxDrawdown().calculate(df_ptf['ptf_creturns'])
        dict_key_metrics['calmar_ratio'] =CalmarRatio(frequency).calculate(df_ptf['ptf_returns'], dict_key_metrics['max_drawdown'])
        try:
            dict_key_metrics['beta']=Beta().calculate(df_ptf_vs_bench['ptf_returns'], df_ptf_vs_bench[f'{symbol}_returns'])
            dict_key_metrics['alpha']=Alpha(frequency, risk_free_rate).calculate(df_ptf_vs_bench['ptf_returns'],
                                                        df_ptf_vs_bench[f'{symbol}_returns'], dict_key_metrics['beta'])
        except Exception as e:
            dict_key_metrics['beta']=0
            dict_key_metrics['alpha']=0

        df_key_metrics=pd.DataFrame.from_dict(dict_key_metrics, orient='index').T

        return df_key_metrics

