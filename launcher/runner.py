import logging
from datetime import timedelta, datetime

import pytz

from backtester.strat_creator import SMAVectorBacktester, MomVectorBacktester, MRVectorBacktester, \
    LRVectorBacktester, ScikitVectorBacktester
from backtester.strat_comparator import StrategyRunner
from backtester.signal_generator import LiveStrategyRunner

if __name__ == '__main__':
    # Main execution logic with comparison of SMA and MOM strategies
    strategies = {
        'SMA': SMAVectorBacktester,
        'MOM': MomVectorBacktester,
        'MeanRev': MRVectorBacktester,
        'LinearReg': LRVectorBacktester}
        # 'ScikitReg': ScikitVectorBacktester}

    param_grids = {
        'SMA': {'sma_short': (5,30), 'sma_long': (31,100)},
        'MOM': {'momentum':(10, 100)},
        'MeanRev': {'sma': (5,50), 'threshold': (0.3,0.7)},
        'LinearReg': {'lags': (3,10), 'train_percent': (0.7, 0.8)}}
        # 'ScikitReg': {'lags': (3, 10), 'train_percent': (0.7, 0.8), 'model': ['logistic']}}

    symbol = 'IBM'
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
    best_strat_recap, comparison_data = runner.run_and_compare_strategies()
    print(best_strat_recap)


    #run live strategies
    data_provider='Quandl'
    trading_platform='binance'
    LiveStrategyRunner(strategies, optimization_results, symbol, start_date, end_date, amount,
                       transaction_costs, data_provider,trading_platform).run()