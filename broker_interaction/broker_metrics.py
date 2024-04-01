import os
import alpaca_trade_api as tradeapi
import pandas as pd
from indicators.performances_indicators import (AnnualizedSharpeRatio, AnnualizedCalmarRatio,
                                                MaxDrawdown, AnnualizedSortinoRatio, Beta,
                                                AnnualizedAlpha, Returns, CumulativeReturns)


class TradingPlatform:
    """Base class for all trading platforms."""

    def api_connection(self):
        raise NotImplementedError

    def get_account_info(self):
        raise NotImplementedError

    def get_all_orders(self):
        raise NotImplementedError

    def get_all_positions(self):
        raise NotImplementedError

    def get_broker_portfolio_history(self):
        raise NotImplementedError

    def get_portfolio_metrics(self, frequency, symbol, risk_free_rate, df_benchmark):
        raise NotImplementedError

class AlpacaPlatform(TradingPlatform):
    """Alpaca trading platform implementation."""
    def __init__(self, config, symbol: str, frequency: dict):
        self.symbol=symbol
        self.frequency=frequency
        self.config=config
        self.get_api_connection()
        self.broker_tracker_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                              f'positions_pnl_tracker/{self.symbol}_{self.frequency['interval']}_broker_ptf_history.csv')

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

    def get_all_orders(self):
        orders=self.api.list_orders(status='all')
        orders_list=[order._raw for order in orders]
        df_orders=pd.DataFrame(orders_list)
        if df_orders.empty:
            return pd.DataFrame()
        else:
            return df_orders[['created_at', 'filled_at', 'asset_id', 'symbol',
                         'asset_class', 'qty', 'filled_qty', 'order_type', 'side', 'filled_avg_price',
                         'time_in_force', 'limit_price', 'stop_price']]

    def get_symbol_orders(self, symbol):
        orders = self.api.list_orders(status='all', symbols=[symbol])
        orders_list = [order._raw for order in orders]
        df_orders = pd.DataFrame(orders_list)
        if df_orders.empty:
            return pd.DataFrame()
        else:
            return df_orders[['created_at', 'filled_at', 'asset_id', 'symbol',
                              'asset_class', 'qty', 'filled_qty', 'order_type', 'side', 'filled_avg_price',
                              'time_in_force', 'limit_price', 'stop_price']]

    def get_all_positions(self):
        positions=self.api.list_positions()
        positions_list=[position._raw for position in positions]
        df_positions=pd.DataFrame(positions_list)
        return df_positions

    def get_symbol_position(self, symbol):
        position=self.api.get_position(symbol)
        pos=pd.DataFrame(pd.Series(position._raw)).T
        return pos

    def get_assets(self):
        assets=self.api.list_assets()
        return assets

    def get_broker_portfolio_history(self):
        portfolio_history=self.api.get_portfolio_history(period='1W', timeframe='1Min', extended_hours=True).df
        return portfolio_history

    def get_all_portfolio_history(self):
        df_ptf_last_day = self.get_broker_portfolio_history()
        df_ptf_last_day.set_index('timestamp', inplace=True)
        df_ptf_last_day = df_ptf_last_day.tz_localize('Europe/Paris')
        try:
            df_ptf_history = pd.read_csv(self.broker_tracker_csv, header=[0], index_col=[0])
            df_ptf_history.index = pd.to_datetime(df_ptf_history.index).tz_convert('Europe/Paris')
            df_ptf = df_ptf_last_day.combine_first(df_ptf_history)
        except Exception as e:
            df_ptf = df_ptf_last_day
        df_ptf.to_csv(self.broker_tracker_csv, mode='w', header=True, index=True)
        return df_ptf



    def get_portfolio_metrics(self, frequency, symbol, risk_free_rate, df_benchmark):
        dict_key_metrics={}
        df_ptf_hist=self.get_all_portfolio_history()
        df_positions=self.get_all_positions()
        df_ptf=df_ptf_hist.join(df_positions)
        (df_ptf['position'].shift(1) * df_ptf['returns']).fillna(0)
        df_ptf=df_ptf.dropna(axis=0)
        try:
            df_benchmark=df_benchmark.tz_localize('Europe/Paris')
        except Exception as e:
            pass
        df_ptf_vs_bench = df_ptf.merge(df_benchmark, left_index=True, right_index=True, how='outer')
        df_ptf_vs_bench=df_ptf_vs_bench.dropna(axis=0)
        # Calculate returns
        df_ptf_vs_bench['ptf_returns'] =Returns().get_metric(df_ptf_vs_bench['position'].shift(1) *
                                                             df_ptf_vs_bench[f'{symbol}_returns']).fillna(0)
        df_ptf_vs_bench['ptf_creturns'] = CumulativeReturns().get_metric(df_ptf_vs_bench['base_value'].iloc[0],
                                                                         df_ptf_vs_bench['ptf_returns'])

        # Calculate Performance Indicators

        try:
            dict_key_metrics['sharpe_ratio'] = AnnualizedSharpeRatio(frequency, risk_free_rate).calculate(df_ptf_vs_bench['ptf_returns'])
        except Exception as e:
            dict_key_metrics['sharpe_ratio'] =0
        dict_key_metrics['sortino_ratio'] =AnnualizedSortinoRatio(frequency, risk_free_rate).calculate(df_ptf_vs_bench['ptf_returns'])

        dict_key_metrics['max_drawdown'] =MaxDrawdown().calculate(df_ptf_vs_bench['ptf_creturns'])
        dict_key_metrics['calmar_ratio'] =AnnualizedCalmarRatio(frequency).calculate(df_ptf_vs_bench['ptf_returns'], dict_key_metrics['max_drawdown'])
        try:
            dict_key_metrics['beta']=Beta().calculate(df_ptf_vs_bench['ptf_returns'], df_ptf_vs_bench[f'{symbol}_returns'])
            dict_key_metrics['alpha']=AnnualizedAlpha(frequency, risk_free_rate).calculate(df_ptf_vs_bench['ptf_returns'],
                                                        df_ptf_vs_bench[f'{symbol}_returns'], dict_key_metrics['beta'])
        except Exception as e:
            dict_key_metrics['beta']=0
            dict_key_metrics['alpha']=0

        df_key_metrics=pd.DataFrame.from_dict(dict_key_metrics, orient='index').T

        return  df_key_metrics

