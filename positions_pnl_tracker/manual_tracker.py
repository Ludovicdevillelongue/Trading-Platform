import pandas as pd
from matplotlib import pyplot as plt

from data_loader.data_retriever import DataManager
from indicators.performances_indicators import Returns, CumulativeReturns, SharpeRatio, SortinoRatio, MaxDrawdown, \
    CalmarRatio, Beta, Alpha


class LiveStrategyTracker():
    def __init__(self, data_provider:str, symbol:str, frequency:dict, start_date, end_date, amount):
        self.data_provider=data_provider
        self.symbol=symbol
        self.frequency=frequency
        self.start_date=start_date
        self.end_date=end_date
        self.amount=amount


    def get_data(self):
        if self.data_provider=='yfinance':
            historical_close_price=pd.DataFrame(DataManager(self.frequency, self.start_date, self.end_date).
                                                yfinance_download(self.symbol)['close'])
        else:
            historical_close_price=pd.DataFrame()
        return historical_close_price

    def get_positions(self):
        positions_recap=pd.read_csv(f'../positions_pnl_tracker/{self.symbol}_position_history.csv')
        positions_recap['time']=pd.to_datetime(positions_recap['time'])
        return (positions_recap.set_index('time')).tz_localize('Europe/Paris')


    def get_asset_history(self):
        positions_recap=self.get_positions()
        historical_close_price=self.get_data()
        asset_history=positions_recap.merge(historical_close_price, left_index=True,
                                             right_index=True, how='outer').ffill(axis=0).dropna()
        asset_history['amount']=asset_history['close']*asset_history['position']
        asset_history['returns']=Returns().get_metric(asset_history['close'])
        asset_history['strategy']=asset_history['position'].shift(1) * asset_history['returns']
        asset_history['cstrategy'] = CumulativeReturns().get_metric(self.amount, asset_history['strategy'])
        asset_history['p&l'] = 0
        for position_date in positions_recap.index:
            mask = asset_history.index >= position_date
            asset_history.loc[mask, 'p&l'] += ((asset_history.loc[mask, 'close'] - asset_history.at[position_date, 'close']) *
                    asset_history.at[position_date, 'order'])
        asset_history=asset_history.dropna(axis=0)
        return asset_history



    def get_asset_metrics(self, frequency, risk_free_rate):
        dict_key_metrics = {}
        df_asset = self.get_asset_history()

        # Calculate Performance Indicators
        try:
            dict_key_metrics['sharpe_ratio'] = SharpeRatio(frequency, risk_free_rate).calculate(df_asset['strategy'])
        except Exception as e:
            dict_key_metrics['sharpe_ratio'] = 0
        dict_key_metrics['sortino_ratio'] = SortinoRatio(frequency, risk_free_rate).calculate(df_asset['strategy'])

        dict_key_metrics['max_drawdown'] = MaxDrawdown().calculate(df_asset['cstrategy'])
        dict_key_metrics['calmar_ratio'] = CalmarRatio(frequency).calculate(df_asset['strategy'],
                                                                            dict_key_metrics['max_drawdown'])
        try:
            dict_key_metrics['beta'] = Beta().calculate(df_asset['strategy'],
                                                        df_asset['returns'])
            dict_key_metrics['alpha'] = Alpha(frequency, risk_free_rate).calculate(df_asset['strategy'],
                                                                                   df_asset['returns'],
                                                                                   dict_key_metrics['beta'])
        except Exception as e:
            dict_key_metrics['beta'] = 0
            dict_key_metrics['alpha'] = 0

        df_key_metrics = pd.DataFrame.from_dict(dict_key_metrics, orient='index').T

        return df_key_metrics



