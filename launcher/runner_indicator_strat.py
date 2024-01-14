import logging
from datetime import timedelta, datetime
import threading
import pytz
import sys
import os
import time
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from broker_interaction.broker_order import GetBrokersConfig
from broker_interaction.broker_metrics import AlpacaPlatform
from positions_pnl_tracker.pnl_tracker_dashboard import PortfolioManagementApp
from backtester.strat_optimizer import RandomSearchAlgorithm, GridSearchAlgorithm, \
    SimulatedAnnealingAlgorithm, GeneticAlgorithm
from backtester.indicator_strat_creator import SMAVectorBacktester, BollingerBandsBacktester, RSIVectorBacktester, \
    MomVectorBacktester, MRVectorBacktester, TurtleVectorBacktester, ParabolicSARBacktester, MACDStrategy, \
    IchimokuStrategy, StochasticOscillatorStrategy, ADXStrategy, VolumeStrategy, WilliamsRBacktester, \
    VolatilityBreakoutBacktester
from backtester.strat_comparator import StrategyRunner
from results_backtest.backtester_dashboard import BacktestApp
from signal_generator.signal_sender import LiveStrategyRunner

if __name__ == '__main__':
    """
    --------------------------------------------------------------------------------------------------------------------
    -----------------------------------------Prepare Inputs and Parameters----------------------------------------------
    --------------------------------------------------------------------------------------------------------------------
    """

    strategies = {
        # 'SMA': SMAVectorBacktester,
        'BB': BollingerBandsBacktester,
        'RSI': RSIVectorBacktester,
        'MOM': MomVectorBacktester,
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
    regression_methods = ['linear', 'poly', 'logistic','ridge', 'lasso', 'elastic_net', 'bayesian', 'svr', 'no_regression']

    param_grids = {
        # 'SMA': {'sma_short': (1, 10), 'sma_long': (10, 30), 'reg_method': regression_methods},
        'BB': {'window_size': (10, 30), 'num_std_dev': (1.0, 2.5), 'reg_method': regression_methods},
        'RSI': {'RSI_period': (5, 15), 'overbought_threshold': (65, 75), 'oversold_threshold': (25, 35), 'reg_method': regression_methods},
        'MOM': {'momentum': (5, 15), 'reg_method': regression_methods},
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

    opti_algo = [RandomSearchAlgorithm(), GridSearchAlgorithm(), SimulatedAnnealingAlgorithm(), GeneticAlgorithm()]
    data_provider = 'yfinance'
    broker_config = GetBrokersConfig.key_secret_tc_url()
    data_freq = os.path.join(os.path.dirname(__file__), '../config/data_frequency.yml')
    with open(data_freq, 'r') as file:
        frequency = yaml.safe_load(file)
    frequency = frequency[data_provider]['minute']
    symbol = 'TSLA'
    start_date = '2023-11-15 00:00:00'
    end_date = ((datetime.now(pytz.timezone('US/Eastern')) - timedelta(minutes=2)).replace(second=0)).strftime(
        "%Y-%m-%d %H:%M:%S")
    invested_amount = 100000
    transaction_costs = broker_config['alpaca']['transaction_costs']
    contract_multiplier = 10
    iterations = 100
    predictive_strat=False

    """
    --------------------------------------------------------------------------------------------------------------------
    -------------------------------------Run Comparison and Optimzation Of Strategies-----------------------------------
    --------------------------------------------------------------------------------------------------------------------
    """
    strat_tester_csv = 'strat_tester_recap.csv'
    with open(strat_tester_csv, 'w') as file:
        pass
    runner = StrategyRunner(strategies, data_provider, frequency, symbol, start_date, end_date,
                            param_grids, opti_algo, invested_amount, transaction_costs, iterations, predictive_strat, strat_tester_csv)
    logging.info("Optimizing strategies...")
    start_time_opti = time.time()
    optimization_results = runner.test_all_search_types()
    end_time_opti = time.time()
    time_diff = end_time_opti - start_time_opti
    print(f'Elapsed time for optimization: {int(time_diff // 60)} minutes '
          f'and {int(time_diff % 60)} seconds')
    logging.info("Optimized results: %s", optimization_results)
    logging.info("\nRunning and comparing strategies...")
    best_strats, comparison_data = runner.run_and_compare_strategies()
    # show results of bactkest in dashboard
    app = BacktestApp(best_strats, comparison_data, symbol)
    threading.Thread(target=app.run_server).start()
    threading.Thread(target=app.open_browser).start()
    best_strat = max(best_strats, key=lambda k: best_strats[k]['results']['sharpe_ratio'])
    print(best_strats)

    """
    --------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------Run Live Strategies-------------------------------------------------
    --------------------------------------------------------------------------------------------------------------------
    """

    threads = []

    trading_platform = 'Alpaca'
    # strat_run = LiveStrategyRunner(best_strat, strategies[best_strat], optimization_results, frequency, symbol,
    #                                start_date, end_date, invested_amount, transaction_costs, predictive_strat,
    #                                contract_multiplier, data_provider, trading_platform, broker_config)
    # trade_sender = threading.Thread(target=strat_run.run)
    # trade_sender.start()
    # threads.append(trade_sender)

    """
    --------------------------------------------------------------------------------------------------------------------
    ------------------------------------------Get Recap From Brokerage Platform-----------------------------------------
    --------------------------------------------------------------------------------------------------------------------
    """

    alpaca = AlpacaPlatform(broker_config)

    portfolio_manager_app = PortfolioManagementApp(alpaca, frequency, symbol, comparison_data['returns'])
    # Start the PortfolioManagementApp server thread
    portfolio_server_thread = threading.Thread(target=portfolio_manager_app.run_server)
    portfolio_server_thread.start()
    threads.append(portfolio_server_thread)

    # Start the PortfolioManagementApp browser thread
    portfolio_browser_thread = threading.Thread(target=portfolio_manager_app.open_browser)
    portfolio_browser_thread.start()
    threads.append(portfolio_browser_thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
