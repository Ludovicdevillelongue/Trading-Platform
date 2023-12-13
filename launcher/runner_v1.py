import logging
from datetime import timedelta, datetime
import threading
import pytz
import sys
import os
from broker_interaction.broker_interaction import GetBrokersConfig
from positions_pnl_tracker.trading_platform_recap import AlpacaPlatform
from positions_pnl_tracker.pnl_tracker_dashboard import PortfolioManagementApp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backtester.strat_optimizer import RandomSearchAlgorithm, GridSearchAlgorithm, \
    SimulatedAnnealingAlgorithm, GeneticAlgorithm
from backtester.strat_creator import SMAVectorBacktester, BollingerBandsBacktester, RSIVectorBacktester, \
    MomVectorBacktester, MRVectorBacktester, TurtleVectorBacktester, ParabolicSARBacktester
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
        'SMA': SMAVectorBacktester,
        'BB': BollingerBandsBacktester,
        'RSI': RSIVectorBacktester,
        'MOM': MomVectorBacktester,
        'MeanRev': MRVectorBacktester,
        'Turtle': TurtleVectorBacktester,
        'ParabolicSAR': ParabolicSARBacktester}
    # 'VolBreakout':VolatilityBreakoutBacktester}
    # 'LinearReg': LRVectorBacktester,
    # 'ScikitReg': ScikitVectorBacktester}

    param_grids = {
        'SMA': {'sma_short': (5, 30), 'sma_long': (31, 100)},
        'BB': {'window_size': (20, 50), 'num_std_dev': (0.5, 2)},
        'RSI': {'RSI_period': (20, 50), 'overbought_threshold': (60, 80), 'oversold_threshold': (20, 40)},
        'MOM': {'momentum': (10, 100)},
        'MeanRev': {'sma': (5, 50), 'threshold': (0.3, 0.7)},
        'Turtle': {'window_size': (20, 50)},
        'ParabolicSAR': {'SAR_step': (0.02, 0.06), 'SAR_max': (0.2, 0.6)}}
    # 'VolBreakout': {'volatility_window': (1, 50), 'breakout_factor': (0.2, 1.4)},
    # 'LinearReg': {'lags': (3,10), 'train_percent': (0.7, 0.8)}}
    # 'ScikitReg': {'lags': (3, 10), 'train_percent': (0.7, 0.8), 'model': ['logistic']}}

    opti_algo=[RandomSearchAlgorithm(), GridSearchAlgorithm(), SimulatedAnnealingAlgorithm(), GeneticAlgorithm()]

    data_provider = 'yfinance'
    symbol = 'TSLA'
    start_date = '2023-11-15 00:00:00'
    end_date = ((datetime.now(pytz.timezone('US/Eastern')) - timedelta(minutes=2)).replace(second=0)).strftime(
        "%Y-%m-%d %H:%M:%S")
    amount = 100000
    transaction_costs = 0.01
    iterations = 100

    """
    --------------------------------------------------------------------------------------------------------------------
    -------------------------------------Run Comparison and Optimzation Of Strategies-----------------------------------
    --------------------------------------------------------------------------------------------------------------------
    """
    runner = StrategyRunner(strategies, data_provider, symbol, start_date, end_date,
                            param_grids, opti_algo, amount, transaction_costs, iterations)
    logging.info("Optimizing strategies...")
    optimization_results = runner.test_all_search_types()
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
    # threads = []

    trading_platform = 'Alpaca'
    broker_config = GetBrokersConfig.key_secret_url()
    strat_run = LiveStrategyRunner(best_strat, strategies[best_strat], optimization_results, symbol, start_date,
                                   end_date, amount,
                                   transaction_costs, data_provider, trading_platform, broker_config).run()
    # trade_sender = threading.Thread(target=strat_run.run)
    # trade_sender.start()
    # threads.append(trade_sender)

    # """
    # --------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------Get Recap From Brokerage Platform-----------------------------------------
    # --------------------------------------------------------------------------------------------------------------------
    # """

    # alpaca = AlpacaPlatform(config)
    #
    # alpaca.get_api_connection()
    # alpaca.get_account_info()
    # alpaca.get_orders()
    # alpaca.get_positions()
    # alpaca.get_assets()
    # alpaca.get_positions_history()

    # portfolio_manager_app = PortfolioManagementApp(alpaca)
    # portfolio_manager_app.run_server()
    # portfolio_manager_app.open_browser()
    # # Start the PortfolioManagementApp server thread
    # portfolio_server_thread = threading.Thread(target=portfolio_manager_app.run_server)
    # portfolio_server_thread.start()
    # threads.append(portfolio_server_thread)
    #
    # # Start the PortfolioManagementApp browser thread
    # portfolio_browser_thread = threading.Thread(target=portfolio_manager_app.open_browser)
    # portfolio_browser_thread.start()
    # threads.append(portfolio_browser_thread)

    # # Wait for all threads to complete
    # for thread in threads:
    #     thread.join()

