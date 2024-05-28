import os
import warnings

warnings.filterwarnings('ignore')
import alpaca_trade_api as tradeapi
import pandas as pd
import yaml
from data_loader.data_retriever import DataManager
from indicators.performances_indicators import (AnnualizedSharpeRatio, AnnualizedCalmarRatio,
                                                MaxDrawdown, AnnualizedSortinoRatio, Beta,
                                                AnnualizedAlpha, Returns, CumulativeReturns, TreynorRatio,
                                                InformationRatio, TrackingError, ValueAtRisk, ConditionalValueAtRisk,
                                                JensensAlpha)


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

    def get_portfolio_metrics(self, risk_free_rate):
        raise NotImplementedError


class AlpacaPlatform(TradingPlatform):
    """Alpaca trading platform implementation."""

    def __init__(self, config, frequency: dict):
        self.frequency = frequency
        self.config = config
        self.get_api_connection()
        self.equity_value_tracker_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                     f'portfolio_manager\\{self.frequency['interval']}_broker_equity_value.csv')
        self.pos_returns_tracker_yml = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                    f'portfolio_manager\\{self.frequency['interval']}_broker_pos_ret.yml')
        self.ptf_metrics_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                            f'portfolio_manager\\{self.frequency['interval']}_broker_ptf_metrics.csv')

    def get_api_connection(self):
        api_key = self.config['alpaca']['api_key']
        api_secret = self.config['alpaca']['api_secret']
        base_url = "https://paper-api.alpaca.markets"
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    def get_account_info(self):
        account = self.api.get_account()
        dict_account_info = {
            'currency': account.currency,
            'pending_transfer_in': account.pending_transfer_in,
            'created_at': account.created_at,
            'position_market_value': account.position_market_value,
            'cash': account.cash,
            'accrued_fees': account.accrued_fees,
            'buying_power': account.buying_power,
            'portfolio_value': account.portfolio_value
        }
        df_account_info = pd.DataFrame.from_dict(dict_account_info, orient='index').T
        return df_account_info

    def get_all_orders(self):
        orders = self.api.list_orders(status='all')
        orders_list = [order._raw for order in orders]
        df_orders = pd.DataFrame(orders_list)
        if df_orders.empty:
            return pd.DataFrame()
        else:
            df_orders['created_at'] = pd.to_datetime(df_orders['created_at']).dt.tz_convert(
                'Europe/Paris')
            df_orders['filled_at'] = pd.to_datetime(df_orders['filled_at']).dt.tz_convert(
                'Europe/Paris')
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
            df_orders['created_at'] = pd.to_datetime(df_orders['created_at']).dt.tz_convert(
                'Europe/Paris')
            df_orders['filled_at'] = pd.to_datetime(df_orders['filled_at']).dt.tz_convert(
                'Europe/Paris')
            return df_orders[['created_at', 'filled_at', 'asset_id', 'symbol',
                              'asset_class', 'qty', 'filled_qty', 'order_type', 'side', 'filled_avg_price',
                              'time_in_force', 'limit_price', 'stop_price']]

    def get_all_positions(self):
        positions = self.api.list_positions()
        positions_list = [position._raw for position in positions]
        df_positions = pd.DataFrame(positions_list)
        return df_positions

    def get_symbol_position(self, symbol):
        symbol = symbol.replace("/", "") if '/' in symbol else symbol
        position = self.api.get_position(symbol)
        pos = pd.DataFrame(pd.Series(position._raw)).T
        pos = pos.round(2)
        return pos

    def get_assets(self):
        assets = self.api.list_assets()
        return assets

    def create_positions_pnl_table(self):
        df_positions = self.get_all_positions()
        try:
            df_positions = df_positions[
                ['symbol', 'current_price', 'qty', 'side', 'market_value', 'unrealized_pl']].round(2)
        except Exception as e:
            df_positions = pd.DataFrame()
        return df_positions

    def create_orders_table(self):
        df_orders = self.get_all_orders()
        df_orders = df_orders[df_orders['filled_at'].notna()]
        df_orders['filled_at'] = pd.to_datetime(df_orders['filled_at'])
        df_orders = df_orders.sort_values(by='filled_at')
        df_orders = df_orders.reset_index(drop=True)

        # Calculate PnL for each order
        df_orders['pnl'] = 0.0
        symbols = df_orders['symbol'].unique()
        for symbol in symbols:
            symbol_orders = df_orders[df_orders['symbol'] == symbol]
            for i in range(1, len(symbol_orders)):
                previous_order = symbol_orders.iloc[i - 1]
                current_order = symbol_orders.iloc[i]
                price_diff = float(current_order['filled_avg_price']) - float(previous_order['filled_avg_price'])
                qty = float(previous_order['filled_qty'])
                if previous_order['side'] == 'buy' and current_order['side'] == 'sell':
                    df_orders.at[i - 1, 'pnl'] = price_diff * qty
                elif previous_order['side'] == 'sell' and current_order['side'] == 'buy':
                    df_orders.at[i - 1, 'pnl'] = -price_diff * qty
        df_orders = df_orders.sort_values('filled_at', ascending=False).round(2)
        return df_orders

    def get_broker_portfolio_history(self):
        portfolio_history = self.api.get_portfolio_history(period='1W', timeframe='1Min', extended_hours=True).df
        return portfolio_history

    def get_all_portfolio_history(self):
        df_ptf_last_day = self.get_broker_portfolio_history()
        df_ptf_last_day = df_ptf_last_day.tz_convert('Europe/Paris')
        try:
            df_ptf_history = pd.read_csv(self.equity_value_tracker_csv, header=[0], index_col=[0])
            df_ptf_history.index = pd.to_datetime(df_ptf_history.index).tz_convert('Europe/Paris')
            df_ptf = df_ptf_last_day.combine_first(df_ptf_history)
        except Exception as e:
            df_ptf = df_ptf_last_day
        df_ptf = df_ptf[df_ptf['equity'] > 100]
        df_ptf.to_csv(self.equity_value_tracker_csv, mode='w', header=True, index=True)
        return df_ptf

    def get_portfolio_returns(self):
        # get symbols previously and currently traded
        positions_alpaca = self.get_all_positions()
        try:
            symbols_currently_traded = [symbol for symbol in positions_alpaca.symbol]
        except Exception as e:
            symbols_currently_traded = []
        # Check if the yml file already exists
        try:
            with open(self.pos_returns_tracker_yml, 'r') as file:
                dict_returns = yaml.safe_load(file)
                symbols_previously_traded = list(set(dict_returns.keys()) - {'portfolio'})

        except FileNotFoundError:
            dict_returns = {}
        try:
            all_symbols = list(set(symbols_previously_traded + symbols_currently_traded))
        except Exception as e:
            all_symbols = symbols_currently_traded
        # Retrieve historical price data for each symbol previously or currently traded,
        # calculate strat return of each asset
        for symbol in all_symbols:
            if symbol == 'BTCUSD':
                symbol_yfinance = 'BTC-USD'
            elif symbol == 'ETHUSD':
                symbol_yfinance = 'ETH-USD'
            else:
                symbol_yfinance = symbol
            df_historical_data = pd.DataFrame(DataManager(self.frequency).yfinance_download(symbol_yfinance).close)
            df_historical_data.index.name = 'timestamp'
            df_historical_data.reset_index(drop=False, inplace=True)

            # Calculate Returns
            df_historical_data['returns'] = Returns().get_metric(df_historical_data['close'])
            # Get the current symbol position
            df_historical_data['position'] = float(0)
            try:
                symbol_position = float(positions_alpaca[positions_alpaca['symbol'] == symbol].qty.values[0])
                # Set the last value of 'position' to symbol_position
                df_historical_data.at[df_historical_data.index[-1], 'position'] = symbol_position
            except Exception as e:
                df_historical_data.at[df_historical_data.index[-1], 'position'] = 0

            #save each symbol data into dict
            try:
                dict_returns[symbol] = pd.DataFrame(dict_returns[symbol]).reset_index(drop=True)
                dict_returns[symbol]['timestamp'] = pd.to_datetime(dict_returns[symbol]['timestamp']).dt.tz_localize(
                    'Europe/Paris')
                dict_returns[symbol] = pd.concat([dict_returns[symbol], df_historical_data.iloc[[-1]]]).sort_values(
                    by='timestamp')
                dict_returns[symbol] = dict_returns[symbol].drop_duplicates(subset=['timestamp'], keep='first')
                dict_returns[symbol]['strategy'] = (
                        dict_returns[symbol]['position'].shift(1) * dict_returns[symbol]['returns']).fillna(0)
            except Exception as e:
                dict_returns[symbol] = df_historical_data.iloc[[-1]]
                dict_returns[symbol]['strategy'] = 0

        # Combine all strategy data into a single DataFrame to calculate the portfolio returns
        all_data = []
        for symbol in all_symbols:
            all_data.append(dict_returns[symbol][['timestamp', 'returns', 'strategy']])

        # Concatenate and group by timestamp to sum the bench and strategy returns
        df_portfolio = pd.concat(all_data).reset_index(drop=True)
        df_portfolio['timestamp'] = pd.to_datetime(df_portfolio['timestamp'])
        portfolio_returns = df_portfolio.groupby('timestamp').sum().reset_index()
        portfolio_returns = portfolio_returns.rename(columns={'returns': 'bench_returns', 'strategy': 'ptf_returns'})

        # Save the new 'portfolio' DataFrame to the dict
        dict_returns['portfolio'] = portfolio_returns

        # convert dict to dataframe for visualisation
        df_combined = dict_returns['portfolio']

        # Merge all DataFrames from dict_returns based on the 'timestamp' column from the 'portfolio' DataFrame
        for key, df in dict_returns.items():
            if key != 'portfolio':
                df_renamed = df.add_suffix(f'_{key}')
                df_renamed = df_renamed.rename(columns={f'timestamp_{key}': 'timestamp'})
                df_combined = df_combined.merge(df_renamed, on='timestamp', how='left')

        def df_to_dict(df):
            return df.to_dict(orient='list')

        with open(self.pos_returns_tracker_yml, 'w') as file:
            for df_key, df_value in dict_returns.items():
                if isinstance(df_value, pd.DataFrame):
                    df_value = df_value[['timestamp'] + [col for col in df_value.columns if col != 'timestamp']]
                    df_value['timestamp'] = df_value['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
                    # Convert DataFrame to dictionary of lists in chunks
                    chunk_size = 1000  # Adjust based on your data size
                    for i in range(0, len(df_value), chunk_size):
                        chunk = df_value.iloc[i:i + chunk_size]
                        yaml.dump({df_key: df_to_dict(chunk)}, file)
        return dict_returns

    def get_portfolio_metrics(self, risk_free_rate, initial_amount):
        df_ptf_returns = self.get_portfolio_returns()['portfolio']
        df_ptf_returns['bench_creturns'] = CumulativeReturns().get_metric(initial_amount,
                                                                          df_ptf_returns['bench_returns'])
        df_ptf_returns['ptf_creturns'] = CumulativeReturns().get_metric(initial_amount, df_ptf_returns['ptf_returns'])

        # Calculate Performance Indicators
        dict_key_metrics = {}
        try:
            dict_key_metrics['sharpe_ratio'] = AnnualizedSharpeRatio(self.frequency, risk_free_rate).calculate(
                df_ptf_returns['ptf_returns'])
        except Exception as e:
            dict_key_metrics['sharpe_ratio'] = 0
        dict_key_metrics['sortino_ratio'] = AnnualizedSortinoRatio(self.frequency, risk_free_rate).calculate(
            df_ptf_returns['ptf_returns'])

        dict_key_metrics['max_drawdown'] = MaxDrawdown().calculate(df_ptf_returns['ptf_creturns'])
        dict_key_metrics['calmar_ratio'] = AnnualizedCalmarRatio(self.frequency).calculate(
            df_ptf_returns['ptf_returns'],
            dict_key_metrics[
                'max_drawdown'])
        try:
            dict_key_metrics['beta'] = Beta().calculate(df_ptf_returns['ptf_returns'], df_ptf_returns['bench_returns'])
            dict_key_metrics['alpha'] = AnnualizedAlpha(self.frequency, risk_free_rate).calculate(
                df_ptf_returns['ptf_returns'],
                df_ptf_returns['bench_returns'], dict_key_metrics['beta'])
        except Exception as e:
            dict_key_metrics['beta'] = 0
            dict_key_metrics['alpha'] = 0

        dict_key_metrics['treynor_ratio'] = TreynorRatio(risk_free_rate).calculate(df_ptf_returns['ptf_returns'],
                                                                                   dict_key_metrics['beta'])
        dict_key_metrics['information_ratio'] = InformationRatio().calculate(df_ptf_returns['ptf_returns'],
                                                                             df_ptf_returns['bench_returns'])
        dict_key_metrics['tracking_error'] = TrackingError().calculate(df_ptf_returns['ptf_returns'],
                                                                       df_ptf_returns['bench_returns'])
        dict_key_metrics['VaR_95%'] = ValueAtRisk.calculate(df_ptf_returns['ptf_returns'], confidence_level=0.95)
        dict_key_metrics['VaR_99%'] = ValueAtRisk.calculate(df_ptf_returns['ptf_returns'], confidence_level=0.99)
        dict_key_metrics['CVaR_95%'] = ConditionalValueAtRisk.calculate(df_ptf_returns['ptf_returns'],
                                                                        confidence_level=0.95)
        dict_key_metrics['CVaR_99%'] = ConditionalValueAtRisk.calculate(df_ptf_returns['ptf_returns'],
                                                                        confidence_level=0.99)
        dict_key_metrics['jensen_alpha'] = JensensAlpha(self.frequency, risk_free_rate).calculate(
            df_ptf_returns['ptf_returns'],
            df_ptf_returns['bench_returns'], dict_key_metrics['beta'])

        df_key_metrics = pd.DataFrame.from_dict(dict_key_metrics, orient='index').T
        df_key_metrics.to_csv(self.ptf_metrics_csv, mode='w', header=True, index=True)
        df_creturns = df_ptf_returns[['timestamp', 'bench_creturns', 'ptf_creturns']]
        df_creturns.set_index('timestamp', inplace=True)
        df_key_metrics = df_key_metrics.round(2)
        return df_creturns, df_key_metrics
