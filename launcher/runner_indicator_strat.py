import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from datetime import timedelta, datetime
import threading
import pytz
import time
import yaml
from indicators.performances_indicators import RiskFreeRate
from broker_interaction.broker_order import GetBrokersConfig
from trading_strategies.strat_optimizer import RandomSearchAlgorithm, GridSearchAlgorithm, \
    SimulatedAnnealingAlgorithm, GeneticAlgorithm
from trading_strategies.indicator_strat_creator import SMAVectorBacktester, BollingerBandsBacktester, RSIVectorBacktester, \
    MomVectorBacktester, MRVectorBacktester, TurtleVectorBacktester, ParabolicSARBacktester, MACDStrategy, \
    IchimokuStrategy, StochasticOscillatorStrategy, ADXStrategy, VolumeStrategy, WilliamsRBacktester, \
    VolatilityBreakoutBacktester
from trading_strategies.strat_comparator import StrategyRunner
from backtester_tracker.backtester_dashboard import BacktestApp
from signal_generator.signal_sender import LiveStrategyRunner

if __name__ == '__main__':
    """
    --------------------------------------------------------------------------------------------------------------------
    -----------------------------------------Prepare Inputs and Parameters----------------------------------------------
    --------------------------------------------------------------------------------------------------------------------
    """

    strategies = {
        'SMA': SMAVectorBacktester,
        # 'BB': BollingerBandsBacktester,
        # 'RSI': RSIVectorBacktester,
        # 'MOM': MomVectorBacktester,
        # 'MeanRev': MRVectorBacktester,
        # 'Turtle': TurtleVectorBacktester,
        # 'ParabolicSAR': ParabolicSARBacktester,
        # 'MACD': MACDStrategy,
        # 'Ichimoku': IchimokuStrategy,
        # 'StochasticOscillator': StochasticOscillatorStrategy,
        # 'ADX': ADXStrategy,
        # 'Volume': VolumeStrategy,
        # 'WilliamR': WilliamsRBacktester,
        # 'VolatilityBreakout': VolatilityBreakoutBacktester
    }
    regression_methods = ['linear']


    param_grids = {
        'SMA': {'sma_short': (1, 10), 'sma_long': (10, 30), 'reg_method': regression_methods},
        # 'BB': {'window_size': (10, 30), 'num_std_dev': (1.0, 2.5), 'reg_method': regression_methods},
        # 'RSI': {'RSI_period': (5, 15), 'overbought_threshold': (65, 75), 'oversold_threshold': (25, 35), 'reg_method': regression_methods},
        # 'MOM': {'momentum': (5, 15), 'reg_method': regression_methods},
        # 'MeanRev': {'sma': (5, 20), 'threshold': (0.01, 0.1), 'reg_method': regression_methods},
        # 'Turtle': {'window_size': (10, 30), 'reg_method': regression_methods},
        # 'ParabolicSAR': {'SAR_step': (0.01, 0.1), 'SAR_max': (0.1, 0.4), 'reg_method': regression_methods},
        # 'MACD': {'short_window': (3, 8), 'long_window': (10, 20), 'signal_window': (3, 9), 'reg_method': regression_methods},
        # 'Ichimoku': {'conversion_line_period': (3, 6), 'base_line_period': (10, 20), 'leading_span_b_period': (20, 40),
        #              'displacement': (10, 20), 'reg_method': regression_methods},
        # 'StochasticOscillator': {'k_window': (5, 15), 'd_window': (2, 5), 'buy_threshold': (15, 30),
        #                          'sell_threshold': (70, 85), 'reg_method': regression_methods},
        # 'ADX': {'adx_period': (5, 15), 'di_period': (5, 15), 'threshold': (15, 25), 'reg_method': regression_methods},
        # 'Volume': {'volume_threshold': (1.0, 4.0), 'volume_window': (2, 30), 'reg_method': regression_methods},
        # 'WilliamR': {'lookback_period': (5, 15), 'overbought': (10, 20), 'oversold': (80, 90), 'reg_method': regression_methods},
        # 'VolatilityBreakout': {'volatility_window':(10,30), 'breakout_factor':(1.0, 2.0), 'reg_method': regression_methods}
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


    symbol = 'BTC-USD'
    risk_free_rate = RiskFreeRate().get_metric()
    start_date = '2023-11-15 00:00:00'
    end_date = ((datetime.now(pytz.timezone('US/Eastern')) - timedelta(minutes=2)).replace(second=0)).strftime(
        "%Y-%m-%d %H:%M:%S")
    invested_amount = 100000
    transaction_costs = broker_config['alpaca']['transaction_costs']
    contract_multiplier = 1
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

