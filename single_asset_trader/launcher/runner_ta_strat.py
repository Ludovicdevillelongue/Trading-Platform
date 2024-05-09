import logging
from datetime import timedelta, datetime
import threading
import pytz
import sys
import os
import time
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from indicators import RiskFreeRate
from broker_interaction.broker_order import GetBrokersConfig
from single_asset_trader.trading_strategies.strat_optimizer import RandomSearchAlgorithm
from single_asset_trader.trading_strategies.ta_strat_creator import BBandsStrategy
from single_asset_trader.trading_strategies.strat_comparator import StrategyRunner
from single_asset_trader.backtester_tracker.backtester_dashboard import BacktestApp
from single_asset_trader.signal_generator.signal_sender import LiveStrategyRunner

if __name__ == '__main__':
    """
    --------------------------------------------------------------------------------------------------------------------
    -----------------------------------------Prepare Inputs and Parameters----------------------------------------------
    --------------------------------------------------------------------------------------------------------------------
    """

    strategies = {
        'BB': BBandsStrategy,
        # 'DEMA': DEMAStrategy,
        # 'EMA': EMAStrategy,
        # 'HTTrendLine': HTTrendlineStrategy,
        # 'KAMA': KAMAStrategy,
        # 'MA':MAStrategy,
        # 'MAMA': MAMAStrategy,
        # # 'MAVP' : MAVPStrategy,
        # 'MIDPOINT' : MIDPOINTStrategy,
        # 'MIDPRICE' : MIDPRICEStrategy,
        # "SAR": SARStrategy,
        # "SAREXT": SAREXTStrategy,
        # "SMA": SMAStrategy,
        # "T3": T3Strategy,
        # "TEMA": TEMAStrategy,
        # "TRIMA": TRIMAStrategy,
        # "WMA":  WMAStrategy,
        # "AD": ADStrategy,
        # "ADOSC": ADOSCStrategy,
        # "OBV": OBVStrategy

    }

    regression_methods = ['linear', 'poly', 'logistic', 'ridge', 'lasso', 'elastic_net', 'bayesian', 'svr',
                          'no_regression']

    param_grids = {
        'BB': {'timeperiod': (10, 30), 'nbdevup': (2, 3), 'nbdevdn': (2, 3), 'reg_method': regression_methods},
        # 'DEMA': {'timeperiod': (10, 30), 'reg_method': regression_methods},
        # 'EMA': {'timeperiod': (10, 30), 'reg_method': regression_methods},
        # 'HTTrendLine': {'reg_method': regression_methods},
        # 'KAMA': {'timeperiod': (10, 30), 'reg_method': regression_methods},
        # 'MA': {'timeperiod': (10, 30), 'ma_type':['SIMPLE', 'EXPONENTIAL'], 'reg_method': regression_methods},
        # 'MAMA': {'fastlimit': (0.4,0.6), 'slowlimit':(0.04,0.06), 'reg_method': regression_methods},
        # # 'MAVP' : {'periods': np.array(np.arange(2, 5), np.arange(6,9)), 'minperiod':(2,10), 'maxperiod':(20,30)},
        # 'MIDPOINT' : {'timeperiod':(10,30), 'reg_method': regression_methods},
        # 'MIDPRICE' : {'timeperiod':(10,30), 'reg_method': regression_methods},
        # 'SAR': {'acceleration':(0.01,0.03), 'maximum':(0.1,0.3), 'reg_method': regression_methods},
        # 'SAREXT': {'start_value':(0.01,0.03), 'offset_on_reverse':(0.01,0.03), 'acceleration_init_long':(0.01,0.03),
        # 'acceleration_long':(0.01,0.03), 'acceleration_max_long':(0.1,0.3), 'acceleration_init_short':(0.01,0.03),
        # 'acceleration_short':(0.01,0.03), 'acceleration_max_short':(0.1,0.3), 'reg_method': regression_methods},
        # 'SMA': {'timeperiod': (10, 30), 'reg_method': regression_methods},
        # 'T3': {'timeperiod': (10, 30), 'volume_factor': (0.7, 0.9), 'reg_method': regression_methods},
        # 'TEMA':{'timeperiod':(10,30), 'reg_method': regression_methods},
        # 'TRIMA':{'timeperiod':(10,30), 'reg_method': regression_methods},
        # 'WMA':{'timeperiod':(10,30),'reg_method': regression_methods},
        # "AD":{'reg_method': regression_methods},
        # "ADOSC":{'fast_period':(6,6), 'slow_period':(13,13), 'reg_method': regression_methods},
        # "OBV":{'reg_method': regression_methods},
    }

    opti_algo = [RandomSearchAlgorithm()]
    data_provider = 'yfinance'
    broker_config = GetBrokersConfig.key_secret_tc_url()
    with open(os.path.join(os.path.dirname(__file__), '../config/data_frequency.yml'), 'r') as file:
        frequency_yaml = yaml.safe_load(file)
    frequency = frequency_yaml[data_provider]['minute']
    with open(os.path.join(os.path.dirname(__file__), '../config/strat_type_pos.yml'), 'r') as file:
        strat_type_pos_yaml = yaml.safe_load(file)
    strat_type_pos = float(strat_type_pos_yaml['long_only'])


    symbol = 'TSLA'
    risk_free_rate = RiskFreeRate().get_metric()
    start_date = '2023-11-15 00:00:00'
    end_date = ((datetime.now(pytz.timezone('US/Eastern')) - timedelta(minutes=2)).replace(second=0)).strftime(
        "%Y-%m-%d %H:%M:%S")
    invested_amount = 100000
    transaction_costs = broker_config['alpaca']['transaction_costs']
    contract_multiplier = 10
    iterations = 10
    predictive_strat=False

    """
    --------------------------------------------------------------------------------------------------------------------
    -------------------------------------Run Comparison and Optimzation Of Strategies-----------------------------------
    --------------------------------------------------------------------------------------------------------------------
    """
    strat_tester_csv=os.path.join(os.path.dirname(__file__), '../backtester_tracker/strat_tester_recap.csv')
    with open(strat_tester_csv, 'w') as file:
        pass
    runner = StrategyRunner(strategies, strat_type_pos, data_provider, frequency, symbol, risk_free_rate, start_date, end_date,
                            param_grids, opti_algo, invested_amount, transaction_costs, iterations, predictive_strat,
                            strat_tester_csv)
    logging.info("Optimizing trading_strategies...")
    start_time_opti = time.time()
    optimization_results = runner.test_all_search_types()
    end_time_opti = time.time()
    time_diff = end_time_opti - start_time_opti
    print(f'Elapsed time for optimization: {int(time_diff // 60)} minutes '
          f'and {int(time_diff % 60)} seconds')
    logging.info("Optimized results: %s", optimization_results)
    logging.info("\nRunning and comparing trading_strategies...")
    best_strats, comparison_data = runner.run_and_compare_strategies()
    # show results of bactkest in dashboard
    app = BacktestApp(best_strats, comparison_data, symbol)
    threading.Thread(target=app.run_server).start()
    threading.Thread(target=app.open_browser).start()
    best_strat = max(best_strats, key=lambda k: best_strats[k]['results']['sharpe_ratio'])

    """
    --------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------Run Live Strategies-------------------------------------------------
    --------------------------------------------------------------------------------------------------------------------
    """

    trading_platform = 'Alpaca'
    strat_run = LiveStrategyRunner(best_strat, strategies[best_strat], strat_type_pos, optimization_results, frequency, symbol,
                                   risk_free_rate,start_date, end_date, invested_amount, transaction_costs, predictive_strat,
                                   contract_multiplier, data_provider, trading_platform, broker_config)
    strat_run.run()

