import logging
from datetime import timedelta, datetime

import pytz

from backtester.strat_creator import SMAVectorBacktester, BollingerBandsBacktester, RSIVectorBacktester
from backtester.strat_comparator import StrategyRunner
from signal_generator.signal_sender import LiveStrategyRunner

if __name__ == '__main__':
    # Main execution logic with comparison of SMA and MOM strategies
    strategies = {
        'SMA': SMAVectorBacktester,
        'BB': BollingerBandsBacktester,
        'RSI': RSIVectorBacktester}
        # 'MOM': MomVectorBacktester,
        # 'MeanRev': MRVectorBacktester,
        # 'Turtle': TurtleVectorBacktester,
        # 'ParabolicSAR': ParabolicSARBacktester,
        # 'VolBreakout':VolatilityBreakoutBacktester}
        # 'LinearReg': LRVectorBacktester,
        # 'ScikitReg': ScikitVectorBacktester}

    param_grids = {
        'SMA': {'sma_short': (5,30), 'sma_long': (31,100)},
        'BB': {'window_size': (20, 50), 'num_std_dev':(0.5, 2)},
        'RSI': {'RSI_period': (20, 50), 'overbought_threshold': (60, 80), 'oversold_threshold': (20, 40)}}
        # 'MOM': {'momentum':(10, 100)},
        # 'MeanRev': {'sma': (5,50), 'threshold': (0.3,0.7)},
        # 'Turtle': {'window_size': (20, 50)},
        # 'ParabolicSAR':{'SAR_step': (0.02, 0.06), 'SAR_max': (0.2, 0.6)},
        # 'VolBreakout': {'volatility_window': (1, 50), 'breakout_factor': (0.2, 1.4)},
        # 'LinearReg': {'lags': (3,10), 'train_percent': (0.7, 0.8)}}
        # 'ScikitReg': {'lags': (3, 10), 'train_percent': (0.7, 0.8), 'model': ['logistic']}}

    symbol = 'TSLA'
    start_date = '2023-11-15 00:00:00'
    end_date = ((datetime.now(pytz.timezone('US/Eastern'))-timedelta(minutes=2)).replace(second=0)).strftime("%Y-%m-%d %H:%M:%S")
    amount = 10000
    transaction_costs = 0.01
    iterations=10

    # Run the comparison and optimization
    runner = StrategyRunner(strategies, symbol, start_date, end_date, param_grids, amount, transaction_costs, iterations)
    logging.info("Optimizing strategies...")
    optimization_results = runner.test_all_search_types()
    logging.info("Optimized results: %s", optimization_results)
    logging.info("\nRunning and comparing strategies...")
    best_parameters_strats, comparison_data = runner.run_and_compare_strategies()
    best_strat = max(best_parameters_strats, key=lambda k: best_parameters_strats[k]['sharpe_ratio'])
    print(best_parameters_strats)


    #run live strategies
    data_provider='Quandl'
    trading_platform='Alpaca'
    LiveStrategyRunner(best_strat, strategies[best_strat], optimization_results, symbol, start_date, end_date, amount,
                       transaction_costs, data_provider,trading_platform).run()