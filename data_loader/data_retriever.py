import os

# import fxcmpy
import quandl as q
import pandas as pd
import yaml
import yfinance as yf
from tables import *

class DataManager(object):
    def __init__(self,frequency, start_date, end_date):
        self.frequency=frequency
        self.start_date=start_date
        self.end_date=end_date

    @property
    def read_key(self):
        config_file = os.path.join(os.path.dirname(__file__), '../config/data_provider_config.yml')
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    @staticmethod
    def convert_datetime(data:pd.DataFrame()):
        data.index = data.index.tz_convert('Europe/Paris')
        return data

    @staticmethod
    def write_data(data:pd.DataFrame()):
        h5 = pd.HDFStore('data.h5', 'w')
        h5['db_name'] = data
        h5.close()

    @staticmethod
    def read_data():
        h5 = pd.HDFStore('data.h5', 'r')
        return h5

    # def fxcm_download(self, symbol):
    #     try:
    #         # Connect to the FXCM API
    #         api_key = self.read_key['fxmcm']['api_key']
    #         con = fxcmpy.fxcmpy(access_token=api_key, server='demo')
    #
    #         # Get the latest data
    #         data_fxcm = con.get_candles(symbol, period='m1',
    #                                       number=1)  # 'm1' for 1-minute data, number=1 for the latest candle
    #         con.close()
    #         return data_fxcm
    #     except Exception as e:
    #         print(f"Error fetching latest data from FXCM: {e}")
    #         return None

    def quandl_download(self, symbol:str):
        data_quandl = q.get(symbol, start_date=self.start_date,
        end_date=self.end_date, api_key=self.read_key['quandl']['api_key'])
        data_quandl.rename(columns={'Value': symbol.split('/')[1]}, inplace=True)
        return data_quandl

    def yfinance_download(self, symbols:list):
        data_yfinance = yf.download(symbols, period=self.frequency['period'], interval=self.frequency['interval'])
        data_yfinance.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close': 'close', 'Volume':'volume'}, inplace=True)
        if self.frequency['interval']=='1m':
            data_yfinance=self.convert_datetime(data_yfinance)
        else:
            pass
        return data_yfinance

