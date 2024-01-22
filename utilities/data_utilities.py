import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import string
import time


def download_ohlcv(crypto_name:str, stake_currency:str, timeframe='1d', since=None, up_to=None,
                   output_dir='../data/garbage/day/', exchange_market='binance'):
    """
        Parameters
        ----------
            crypto_name : str -> name of cryptocurrency to retrieve
            stake_currency : str -> currency chosen as cash
            timeframe : str -> frequency of results (1h, 1d, 1M)
            since: int -> beginning date for data retrieval
            up_to : int -> end date for data retrieval
            output_dir: str -> path where the data will be stored
            exchange_market : str -> exchange used
    """

    #download open high low close of selected crypto on a specific timeframe from a specific data provider
    if since is None:
        since = int(datetime.strptime("01.01.2021", "%d.%m.%Y").timestamp())
    if up_to is None:
        up_to = int(datetime.now().timestamp())
    delta = timedelta(seconds=up_to - since)
    if timeframe == '1m':
        limit = int(delta.total_seconds() / 60)
    elif timeframe == '1h':
        limit = int(delta.total_seconds() / 3600)
    elif timeframe == '1d':
        limit = delta.days
    elif timeframe == '1M':
        dt_since = datetime.fromtimestamp(since)
        dt_up_to = datetime.fromtimestamp(up_to)
        nb_months = (dt_up_to.year - dt_since.year) * 12 + (dt_up_to.month - dt_since.month)
        limit = nb_months
    else:
        raise Exception('Wrong timeframe used to fetch ohlcv data. Possible values are 1m, 1h, 1d, 1M.')

    formatted_pair = '{c1}/{c2}'.format(c1=crypto_name, c2=stake_currency)
    if exchange_market == 'binance':
        exchange = ccxt.binance()
    elif exchange_market == 'bitrue':
        exchange = ccxt.bitrue()
    elif exchange_market == 'bittrex':
        exchange = ccxt.bittrex()
    elif exchange_market == 'huobi':
        exchange = ccxt.huobi()
    elif exchange_market == 'coinbase':
        exchange = ccxt.coinbase()
    elif exchange_market == 'idex':
        exchange = ccxt.idex()
    # since argument must be a timestamp in milliseconds
    candles = {}
    while limit > 1000:
        t = {'1m': 'minutes',
             '1h': 'hours',
             '1d': 'days',
             '1M': 'months'}
        result = exchange.fetch_ohlcv(formatted_pair, timeframe=timeframe, since=since * 1000, limit=1000)
        for candle in result:
            candles[candle[0]] = candle
        limit -= 1000
        since = int(datetime.timestamp(datetime.fromtimestamp(since) + timedelta(**{t[timeframe]: 1000})))
    for candle in exchange.fetch_ohlcv(formatted_pair, timeframe=timeframe, since=since * 1000, limit=limit):
        candles[candle[0]] = candle
    data = list(candles.values())  # Modifies input
    for data_line in data:
        timestamp = data_line[0] / 1000
        formatted_date = datetime.fromtimestamp(timestamp)
        data_line[0] = formatted_date
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    res = pd.DataFrame(data, columns=columns)
    data_path = output_dir + '{c1}_{c2}.csv'.format(c1=crypto_name, c2=stake_currency)
    res.to_csv(data_path, index=False)


def download_cryptos(stake_currency:str, start_date:str, end_date:str, universe=None):
    """
        Parameters
        ----------
            stake_currency : str -> currency chosen as cash
            start_date : str -> beginning date for data retrieval
            end_date : str -> end date for data retrieval
            universe : list -> list of crypto for a category
    """
    start_date = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_date = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    for crypto_name in universe:
        download_ohlcv(crypto_name, stake_currency, timeframe='1d', since=start_date,
                       up_to=end_date)


def check_month_start(day_counter:int, data_handler:classmethod)-> bool:
    """
        Parameters
        ----------
            day_counter : int -> number of days from initial date
            data_handler:classmethod-> data handling
        Returns
        -------
            true or false according to the date
    """
    #check the gap between a date and the start of its associated month
    np_date = data_handler.dates[day_counter]
    pd_date = pd.Timestamp(np_date)
    return pd_date.is_month_start


def check_week_start(day_counter:int, data_handler:classmethod, day=0)-> bool:
    """
        Parameters
        ----------
            day_counter : int -> number of days from initial date
            data_handler:classmethod-> data handling
            day: int -> day in number
        Returns
        -------
            true or false according to the date
    """

    #check the gap between a date and the start of its associated week
    np_date = data_handler.dates[day_counter]
    pd_date = pd.Timestamp(np_date)
    return pd_date.weekday() == day

def request_market_cap(crypto_name:str,start_date:str)-> pd.DataFrame:
    """
        Parameters
        ----------
            crypto_name : str -> name of cryptocurrency to retrieve
            start_date : str -> beginning date for data retrieval
        Returns
        -------
            historical market cap of chosen crypto
    """
    element = datetime.strptime(start_date, '%Y-%m-%d')
    start = int(datetime.timestamp(element))
    end = int(datetime.timestamp(datetime.now()))

    #connexion to coin 360
    coin360_session = Session()
    headers = {
        'Accepts': 'application/json',
    }
    coin360_session.headers.update(headers)

    def call_coin360_url(url:str, params=None)-> json:
        """
            Parameters
            ----------
                url: str -> url of coin360 api
                params:str -> parameters needed for session connection
            Returns
            -------
                json with coin360 response
        """
        try:
            response = coin360_session.get(url, params=params)
            return json.loads(response.text)
        except (ConnectionError, Timeout, TooManyRedirects) as e:
            print(e)

    info_currency = call_coin360_url('https://api.coin360.com/info/currency')

    info_currency_names = {k['name'].translate({ord(c): None for c in string.whitespace}).lower(): k for k in
                           info_currency}
    info_currency_sym = {k['symbol']: k for k in info_currency}
    df_sym = pd.DataFrame.from_dict(info_currency_sym, orient='index')
    df_sym=df_sym.reset_index()
    df_sym=df_sym[df_sym["symbol"]==crypto_name]

    coin360_dict = {}
    excluded = {'SAND'
        , 'STX'
        , 'OMG'
        , 'ANC'
        , 'IMX'
        , 'UOS'
        , 'WIN'
        , 'SUPER'
        , 'ASTR'
        , 'BEST'
        , 'UFO'
        , 'ANY'
        , 'WILD'
        , 'DAR'
        , 'MIR'
        , 'BIFI'
        , 'GTC'
        , 'LINA'
        , 'FLM'
        , 'CUBE'
        , 'FARM'
        , 'DDX'
        , 'SDAO'
        , 'TIME'
        , 'POLIS'
        , 'STT'
        , 'SWTH'
        , 'INV'
        , 'FOR'
        , 'PNT'
        , 'UST'
        , 'LN'
        , 'XPR'
        , 'DIA'
        , 'WHALE'
        , 'HIVE'
        , 'GALA'
        , 'MFG'
        , 'MCB'
        , 'BNC'
        , 'RING'
        , 'ICE'
        , 'LOTTO'
        , 'CRU'
        , 'STRONG'
        , 'EPIK'
        , 'VID'
        , 'VIDT'
        , 'AUTO'
        , 'GET'
        , 'DREP'
        , 'RSV'
        , 'GXT', 'FLOW'}
    not_found = []
    for index, row in df_sym.iterrows():
        if row.symbol in excluded:
            continue
        coin360_coin = info_currency_names.get(row['name'].translate({ord(c): None for c in string.whitespace}).lower())
        if coin360_coin is None:
            coin360_coin = info_currency_sym.get(row.symbol)
        if coin360_coin:
            sym = coin360_coin['symbol']
            params = {'coin': sym, 'start': start, 'end': end, 'period': '1d'}
            res = call_coin360_url('https://api.coin360.com/coin/historical', params=params)
            coin360_dict[row.symbol] = res
            time.sleep(1)
        else:
            not_found.append((row['name'], row.symbol))
    coin360_dict = {k: v for k, v in coin360_dict.items() if len(v)}
    data = []
    for k, v in coin360_dict.items():
        for d in v:
            d['symbol'] = k
            data.append(d)
    market_cap_df = pd.DataFrame(data)
    return market_cap_df



def fetch_market_cap(crypto_name:str, start_day:str, end_day:str)-> np.array:
    """
        Parameters
        ----------
            crypto_name : str -> name of cryptocurrency to retrieve
            start_day : str -> first date from which market cap are retrieved
            end_day : str -> last date from which market cap are retrieved

        Returns
        ---------
            formatted data
    """
    df = request_market_cap(crypto_name,start_day)

    def get_crypto_data(crypto_name:str)-> pd.DataFrame:
        """
            Parameters
            ----------
                crypto_name: str -> name of cryptocurrency to retrieve
            Returns
            -------
                data in reverse order
        """
        return df.loc[df['symbol'] == crypto_name].iloc[::-1]

    def to_datetime(date:str)-> datetime:
        """
            Parameters
            ----------
                date: str -> date to convert
            Returns
            -----------
                date in datetime format
        """
        date_time_obj = datetime.strptime(date, '%Y-%m-%d')
        return date_time_obj

    start_dt_obj = to_datetime(start_day)
    end_dt_obj = to_datetime(end_day)

    df_crypto = get_crypto_data(crypto_name)
    start_tmstmp = int(start_dt_obj.timestamp())
    end_tmstmp = int(end_dt_obj.timestamp())
    df_filtered = df_crypto.loc[(df_crypto['timestamp'] >= start_tmstmp) & (df_crypto['timestamp'] <= end_tmstmp)]

    res = df_filtered['market_cap'].to_numpy()
    return res

