import os

import fxcmpy
import quandl as q
import pandas as pd
import yaml
import yfinance as yf
from tables import *

class DataRetriever(object):
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
    def write_data(data:pd.DataFrame()):
        h5 = pd.HDFStore('data.h5', 'w')
        h5['db_name'] = data
        h5.close()

    @staticmethod
    def read_data():
        h5 = pd.HDFStore('data.h5', 'r')
        return h5

    def fxcm_download(self):
        data_fxcm = self.read_key['fxmcm']['api_key']

    def quandl_download(self, symbol:str):
        data_quandl = q.get(symbol, start_date=self.start_date,
        end_date=self.end_date, api_key=self.read_key['quandl']['api_key'])
        data_quandl.rename(columns={'Value': symbol.split('/')[1]}, inplace=True)
        return data_quandl

    def yfinance_download(self, symbol:str):

        data_yfinance = yf.download(symbol, period=self.frequency['period'], interval=self.frequency['interval'])
        data_yfinance.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close': 'close', 'Volume':'volume'}, inplace=True)
        return data_yfinance

    def quandl_latest_data(self, symbol: str):
        try:
            # Fetching only the latest data by setting rows to 1
            latest_data = q.get(symbol, rows=1, api_key=self.read_key.get('api_keys', 'quandl_api_key'))
            latest_data.rename(columns={'Value': symbol.split('/')[1]}, inplace=True)
            return latest_data
        except Exception as e:
            print(f"Error fetching latest data from Quandl: {e}")
            return None


    def fxcm_latest_data(self, symbol: str):
        try:
            # Connect to the FXCM API
            api_key = self.read_key['fxmcm']['api_key']
            con = fxcmpy.fxcmpy(access_token=api_key, server='demo')

            # Get the latest data
            latest_data = con.get_candles(symbol, period='m1', number=1)  # 'm1' for 1-minute data, number=1 for the latest candle
            con.close()
            return latest_data
        except Exception as e:
            print(f"Error fetching latest data from FXCM: {e}")
            return None

    def yfinance_latest_data(self, symbol:str):
        latest_data = yf.download(symbol, period=self.frequency['period'], interval=self.frequency['interval'])
        latest_data.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close': 'close','Volume':'volume'}, inplace=True)
        return latest_data

