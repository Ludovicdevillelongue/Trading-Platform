import alpaca_trade_api as tradeapi
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

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

    def get_portfolio_metrics(self, frequency, symbol, df_benchmark):
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
        return portfolio_history


    def close_open_positions(self):
        closed_positions=self.api.close_open_positions()
        return closed_positions

    def get_risk_free_rate(self):
        tbill = yf.Ticker("^IRX")  # ^IRX is the symbol for 13-week Treasury Bill
        hist = tbill.history(period="1mo")
        risk_free_rate = hist['Close'].iloc[-1] / 100
        return risk_free_rate


    def get_portfolio_metrics(self, frequency, symbol, df_benchmark):
        dict_key_metrics={}
        df_ptf=self.get_portfolio_history()
        df_ptf.set_index('timestamp', inplace=True)
        df_ptf=df_ptf.tz_localize('America/New_York')
        try:
            df_benchmark=df_benchmark.tz_localize('America/New_York')
        except Exception as e:
            pass
        # Calculate returns
        df_ptf['ptf_returns'] =np.log(df_ptf['equity'] / df_ptf['equity'].shift(1))
        df_ptf['ptf_creturns'] = df_ptf['base_value'].iloc[0] * df_ptf['ptf_returns'].cumsum().apply(np.exp)
        df_ptf=df_ptf.dropna(axis=0)
        df_ptf_vs_bench = df_ptf.merge(df_benchmark, left_index=True, right_index=True, how='outer')
        df_ptf_vs_bench=df_ptf_vs_bench.dropna(axis=0)

        # Calculate the Sharpe Ratio
        risk_free_rate = self.get_risk_free_rate()
        try:
             # Risk-free rate of return
            df_ptf['ptf_excess_returns'] = df_ptf['ptf_returns'] -  (risk_free_rate/ frequency['annualized_coefficient'])
            dict_key_metrics['sharpe_ratio'] = (df_ptf['ptf_excess_returns'].mean() / df_ptf['ptf_excess_returns'].std()) \
                                               * np.sqrt(frequency['annualized_coefficient'])
        except Exception as e:
            sharpe_ratio=0

        # Sortino Ratio
        negative_std = np.std(df_ptf.loc[df_ptf['ptf_returns'] < 0, 'ptf_returns']) * np.sqrt(frequency['annualized_coefficient'])
        dict_key_metrics['sortino_ratio'] = (df_ptf['ptf_returns'].mean() -  (risk_free_rate /
                         frequency['annualized_coefficient'])) / negative_std if negative_std != 0 else 0


        # Drawdown
        roll_max = df_ptf['ptf_creturns'].cummax()
        drawdown = df_ptf['ptf_creturns'] / roll_max - 1.0
        df_ptf['drawdown'] = drawdown

        # Maximum Drawdown
        dict_key_metrics['max_drawdown'] = drawdown.min()

        # Calmar Ratio
        dict_key_metrics['calmar_ratio'] = df_ptf['ptf_returns'].mean() * frequency['annualized_coefficient']  / \
                                           abs(dict_key_metrics['max_drawdown']) if dict_key_metrics['max_drawdown'] != 0 else 0

        # Alpha and Beta (compared to 'df_ptf['return']' as the benchmark)
        try:
            covariance = np.cov(df_ptf_vs_bench['ptf_returns'], df_ptf_vs_bench[f'{symbol}_returns'])[0][1]
            dict_key_metrics['beta'] = covariance / np.var(df_ptf_vs_bench[f'{symbol}_returns'])
            dict_key_metrics['alpha'] = (df_ptf_vs_bench['returns'].mean() - (risk_free_rate / frequency['annualized_coefficient'])) - \
                                        dict_key_metrics['beta'] * \
                                        (df_ptf_vs_bench[f'{symbol}_returns'].mean() -(risk_free_rate / frequency['annualized_coefficient']))
        except Exception as e:
            dict_key_metrics['beta']=0
            dict_key_metrics['alpha']=0

        df_key_metrics=pd.DataFrame.from_dict(dict_key_metrics, orient='index').T

        return df_key_metrics

