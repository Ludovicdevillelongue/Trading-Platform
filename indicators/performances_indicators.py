import numpy as np
import yfinance as yf


class RiskFreeRate():
    @staticmethod
    def get_risk_free_rate():
        tbill = yf.Ticker("^IRX")  # ^IRX is the symbol for 13-week Treasury Bill
        hist = tbill.history(period="1mo")
        risk_free_rate = hist['Close'].iloc[-1] / 100
        return risk_free_rate

class SharpeRatio():
    def __init__(self, frequency, risk_free_rate):
        self.frequency=frequency
        self.risk_free_rate = risk_free_rate

    def calculate(self, returns):
        excess_returns = returns - (self.risk_free_rate / self.frequency['annualized_coefficient'])

        # Assuming 'returns' are not already annualized:
        annualized_mean_excess_return = ((1 + np.mean(excess_returns)) ** self.frequency['annualized_coefficient']) - 1
        annualized_std_excess_return = np.std(excess_returns) * np.sqrt(self.frequency['annualized_coefficient'])

        return annualized_mean_excess_return / annualized_std_excess_return if annualized_std_excess_return != 0 else 0


# Class for Sortino Ratio
class SortinoRatio():
    def __init__(self, frequency, risk_free_rate):
        self.frequency=frequency
        self.risk_free_rate = risk_free_rate

    def calculate(self, returns):
        excess_returns = returns - (self.risk_free_rate / self.frequency['annualized_coefficient'])
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(self.frequency['annualized_coefficient'])
        return (np.mean(excess_returns) / downside_deviation) if downside_deviation != 0 else 0

# Class for Calmar Ratio
class CalmarRatio():
    def __init__(self, frequency):
        self.frequency=frequency

    def calculate(self, returns, max_drawdown):
        mean_return = np.mean(returns)
        annualized_return = (1 + mean_return) ** self.frequency['annualized_coefficient'] - 1
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
        self.frequency=frequency
        self.risk_free_rate = risk_free_rate

    def calculate(self, portfolio_returns, benchmark_returns, beta):
        # Annualizing the mean returns
        annualized_portfolio_return = (1 + np.mean(portfolio_returns)) ** self.frequency['annualized_coefficient'] - 1
        annualized_benchmark_return = (1 + np.mean(benchmark_returns)) ** self.frequency['annualized_coefficient'] - 1

        # Annualized excess returns
        excess_portfolio_return = annualized_portfolio_return - self.risk_free_rate
        excess_benchmark_return = annualized_benchmark_return - self.risk_free_rate

        # Alpha calculation
        alpha = excess_portfolio_return - beta * excess_benchmark_return
        if np.isnan(alpha):
            return 0
        else:
            return alpha

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