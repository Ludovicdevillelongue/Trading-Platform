import os
import pandas as pd
from indicators.performances_indicators import (Returns, CumulativeReturns, AnnualizedSharpeRatio,
                        AnnualizedSortinoRatio, MaxDrawdown,
                        AnnualizedCalmarRatio, Beta, AnnualizedAlpha)
import pytz

class LiveStrategyTracker():
    def __init__(self, data_provider: str, symbol: str, frequency: dict, start_date, end_date, amount):
        self.data_provider = data_provider
        self.symbol = symbol
        self.frequency = frequency
        self.start_date = start_date
        self.end_date = end_date
        self.amount = amount
        self.strat_pos_tracker_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                              f'positions_pnl_tracker/{self.symbol}_{self.frequency['interval']}_strat_positions_history.csv')
        self.strat_equity_ret_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                              f'positions_pnl_tracker/{self.symbol}_{self.frequency['interval']}_strat_history.csv')
        self.strat_metrics_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                              f'positions_pnl_tracker/{self.symbol}_{self.frequency['interval']}_strat_metric.csv')
    def get_previous_positions(self):
        if not os.path.exists(self.strat_pos_tracker_csv) or \
                os.path.getsize(self.strat_pos_tracker_csv) == 0:
            previous_asset_positions = pd.DataFrame()

        else:
            previous_asset_positions = pd.read_csv(self.strat_pos_tracker_csv, header=[0], index_col=[0])
            previous_asset_positions.index = pd.DatetimeIndex(pd.to_datetime
                                                              (previous_asset_positions.index, utc=True))
            previous_asset_positions.index = (previous_asset_positions.
                                              index.tz_convert(pytz.timezone('Europe/Paris')))
        return previous_asset_positions.sort_index()

    def get_asset_history(self, historical_close_price, new_position):
        previous_asset_positions = self.get_previous_positions()
        try:
            new_position['time'] = historical_close_price.index[-1]
            new_position.set_index('time', inplace=True)
        #position is none
        except Exception as e:
            new_position=pd.DataFrame()
        try:
            last_pos=pd.DataFrame(previous_asset_positions[['symbol', 'position', 'order']].iloc[-1]).T
            last_new_pos=pd.concat([last_pos, new_position], axis=0)
        #no record of last position
        except Exception as e:
            last_new_pos=pd.DataFrame([{'symbol': self.symbol,'position': 0,'order': 0}])
        last_new_pos_close=last_new_pos.merge(historical_close_price, left_index=True,
                                              right_index=True, how='outer').ffill().dropna(axis=0)
        total_asset_history=pd.concat([last_new_pos_close, previous_asset_positions], axis=0)
        total_asset_history=(total_asset_history[~total_asset_history.index.duplicated(keep='first')].drop_duplicates()).sort_index()
        try:
            total_asset_history['position']=total_asset_history['position'].round(9)
        except Exception as e:
            pass
        total_asset_history = total_asset_history.fillna(0)
        total_asset_history.to_csv(self.strat_pos_tracker_csv, mode='w', header=True, index=True)
        total_asset_history['amount'] = total_asset_history['close'] * total_asset_history['position']
        total_asset_history['returns'] = Returns().get_metric(total_asset_history['close'])
        total_asset_history['creturns'] = CumulativeReturns().get_metric(1, total_asset_history['returns'])
        total_asset_history['strategy'] = (total_asset_history['position'].shift(1) * total_asset_history['returns']).fillna(0)
        total_asset_history['cstrategy'] = CumulativeReturns().get_metric(1, total_asset_history['strategy'])
        total_asset_history['pnl'] = 0
        # add pnl of last position to historical pnl
        position_changes = total_asset_history['position'].diff().fillna(1) != 0
        position_changes.iloc[0] = True
        for position_date in position_changes[position_changes].index:
            mask = total_asset_history.index >= position_date
            total_asset_history.loc[mask, 'pnl'] += ((total_asset_history.loc[mask, 'close'] -
                                                      total_asset_history.at[position_date, 'close']) *
                                                     total_asset_history.at[position_date, 'order'])

        total_asset_history['product_value']= total_asset_history['pnl']
        total_asset_history.dropna(axis=0)
        total_asset_history.to_csv(self.strat_equity_ret_csv, mode='w', header=True, index=True)
        return total_asset_history

    def get_asset_metrics(self, frequency, risk_free_rate,historical_close_price, new_position):
        dict_key_metrics = {}
        df_asset = self.get_asset_history(historical_close_price, new_position)

        # Calculate Performance Indicators
        try:
            dict_key_metrics['sharpe_ratio'] = AnnualizedSharpeRatio(frequency, risk_free_rate).calculate(df_asset['strategy'])
        except Exception as e:
            dict_key_metrics['sharpe_ratio'] = 0
        dict_key_metrics['sortino_ratio'] = AnnualizedSortinoRatio(frequency, risk_free_rate).calculate(df_asset['strategy'])

        dict_key_metrics['max_drawdown'] = MaxDrawdown().calculate(df_asset['cstrategy'])
        dict_key_metrics['calmar_ratio'] = AnnualizedCalmarRatio(frequency).calculate(df_asset['strategy'],
                                                                            dict_key_metrics['max_drawdown'])
        try:
            dict_key_metrics['beta'] = Beta().calculate(df_asset['strategy'],
                                                        df_asset['returns'])
            dict_key_metrics['alpha'] = AnnualizedAlpha(frequency, risk_free_rate).calculate(df_asset['strategy'],
                                                                                   df_asset['returns'],
                                                                                   dict_key_metrics['beta'])
        except Exception as e:
            dict_key_metrics['beta'] = 0
            dict_key_metrics['alpha'] = 0

        df_key_metrics = pd.DataFrame.from_dict(dict_key_metrics, orient='index').T
        df_key_metrics.to_csv(self.strat_metrics_csv, mode='w', header=True, index=True)
        return df_key_metrics
