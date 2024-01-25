import numpy as np
import pandas as pd
import yfinance as yf


class RiskFreeRate():
    @staticmethod
    def get_metric():
        tbill = yf.Ticker("^IRX")  # ^IRX is the symbol for 13-week Treasury Bill
        hist = tbill.history(period="1mo")
        risk_free_rate = hist['Close'].iloc[-1] / 100
        return risk_free_rate

class Returns:
    @staticmethod
    def get_metric(c: pd.Series) -> np.array:
        res = np.empty_like(c)
        res[0] = 0
        res[1:] = (c.values[1:] - c.values[:-1]) / c.values[:-1]

        return res

class CumulativeReturns():
    @staticmethod
    def get_metric(amount, returns_array: np.array) -> np.array:
        # Convert simple returns to growth factors (1 + return)
        growth_factors = 1 + returns_array
        # Calculate the cumulative product of the growth factors
        cumulative_growth_factors = growth_factors.cumprod()
        # The cumulative value is the initial amount times the cumulative growth factors
        cumulative_values = amount * cumulative_growth_factors
        return cumulative_values

class LogReturns():
    @staticmethod
    def get_metric(returns_array:np.array)-> np.array:
        ones = np.ones_like(returns_array)
        return np.log(returns_array + ones)

class CumulativeLogReturns():
    @staticmethod
    def get_metric(amount, log_returns_array:np.array)-> np.array:
        return amount * (log_returns_array).cumsum().apply(np.exp)

class Vol():
    @staticmethod
    def get_metric(log_returns_array:np.array)-> np.array:
        vol = np.std(log_returns_array)
        return vol

class SharpeRatio():
    def __init__(self, frequency, risk_free_rate):
        self.frequency = frequency
        self.risk_free_rate = risk_free_rate

    def calculate(self, returns):
        excess_returns = returns - (self.risk_free_rate / self.frequency['annualized_coefficient'])
        annualized_mean_excess_return = np.mean(excess_returns) * self.frequency['annualized_coefficient']
        annualized_std_return = np.std(returns) * np.sqrt(self.frequency['annualized_coefficient'])

        return annualized_mean_excess_return / annualized_std_return if annualized_std_return != 0 else 0


# Class for Sortino Ratio
class SortinoRatio():
    def __init__(self, frequency, risk_free_rate):
        self.frequency = frequency
        self.risk_free_rate = risk_free_rate

    def calculate(self, returns):
        # Setting the target return, typically 0 or the risk-free rate
        target_return = self.risk_free_rate / self.frequency['annualized_coefficient']
        downside_returns = returns[returns < target_return]
        # Annualizing the downside deviation
        annualized_downside_deviation = np.std(downside_returns) * np.sqrt(self.frequency['annualized_coefficient'])
        excess_returns = returns - (self.risk_free_rate / self.frequency['annualized_coefficient'])
        annualized_mean_excess_return = np.mean(excess_returns) * self.frequency['annualized_coefficient']
        # Calculating the Sortino Ratio using the annualized downside deviation
        return annualized_mean_excess_return / annualized_downside_deviation if annualized_downside_deviation != 0 else 0


# Class for Calmar Ratio
class CalmarRatio():
    def __init__(self, frequency):
        self.frequency = frequency

    def calculate(self, returns, max_drawdown):
        annualized_return = ((1 + np.mean(returns)) ** self.frequency['annualized_coefficient']) - 1
        return annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

# Class for Maximum Drawdown
class MaxDrawdown:
    @staticmethod
    def calculate(cumulative_returns):
        roll_max = cumulative_returns.cummax()
        drawdown = cumulative_returns / roll_max - 1.0
        return drawdown.min()

# Class for Alpha
class Alpha():
    def __init__(self, frequency, risk_free_rate):
        self.frequency = frequency
        self.risk_free_rate = risk_free_rate

    def calculate(self, portfolio_returns, benchmark_returns, beta):
        annualized_portfolio_return = ((1 + np.mean(portfolio_returns)) ** self.frequency['annualized_coefficient']) - 1
        annualized_benchmark_return = ((1 + np.mean(benchmark_returns)) ** self.frequency['annualized_coefficient']) - 1

        excess_portfolio_return = annualized_portfolio_return - self.risk_free_rate
        excess_benchmark_return = annualized_benchmark_return - self.risk_free_rate

        alpha = excess_portfolio_return - beta * excess_benchmark_return
        return alpha if not np.isnan(alpha) else 0

# Class for Beta
class Beta:
    @staticmethod
    def calculate(portfolio_returns, benchmark_returns):
        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
        variance = np.var(benchmark_returns)
        beta=covariance / variance if variance != 0 else 0
        if np.isnan(beta):
            return 0
        else:
            return beta