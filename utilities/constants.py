from indicators.performances_indicators import *

#Metrics to retrieve
ALL_PRINT_INDICATORS = [
        cumulative_returns,
        daily_returns,
        annual_returns,
        annual_volatility,
        value_at_risk,
        expected_shortfall,
        drawdown,
        max_drawdown,
        sharpe_ratio,
        calmar_ratio,
        alpha,
        beta
    ]

#Day of week conversion
DAY_DCT = {
    'monday': 0,
    'tuesday': 1,
    'wednesday': 2,
    'thursday': 3,
    'friday': 4,
    'saturday': 5,
    'sunday': 6
}