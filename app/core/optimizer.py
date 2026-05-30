import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import linregress

class PortfolioOptimizer:
    
    def __init__(self, risk_free_rate: float = 0.0):
        # Risk-free rate is used when calculating the Sharpe ratio.
        # Defaults to 0 if not provided — caller can pass in the current T-bill rate for more accuracy.
        self.risk_free_rate = risk_free_rate
    
    def calculate_portfolio_returns(self, weights: np.ndarray, returns: pd.DataFrame) -> np.ndarray:
        # Matrix multiply asset returns by weights to get the portfolio's daily return series.
        # This is the building block that almost every other method in this class calls first.
        return returns @ weights
    
    def calculate_portfolio_volatility(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        # Standard deviation of daily returns scaled up to annual by multiplying by sqrt(252),
        # since there are roughly 252 trading days in a year.
        portfolio_returns = self.calculate_portfolio_returns(weights, returns)
        return portfolio_returns.std() * np.sqrt(252)
    
    def calculate_portfolio_sharpe(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        # Sharpe ratio = annualized excess return divided by annualized volatility.
        # We subtract the daily risk-free rate (annual rate / 252) from each day's return
        # before computing the ratio. Guard against division by zero in case volatility is flat.
        portfolio_returns = self.calculate_portfolio_returns(weights, returns)
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
    
    def calculate_max_drawdown(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        # Convert daily returns to a cumulative growth curve, then at each point find how far
        # we've fallen from the highest value seen so far. The worst such drop is the max drawdown.
        # We return it as a positive percentage (e.g. 15.3 means the portfolio fell 15.3% peak-to-trough).
        portfolio_returns = self.calculate_portfolio_returns(weights, returns)
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min()) * 100
    
    def calculate_cagr(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        # Compound the daily returns into a single total return, then annualize it.
        # The formula (1 + total_return)^(1/years) - 1 smooths out the return as if
        # it grew at a steady rate every year — that's what "compound annual" means.
        portfolio_returns = self.calculate_portfolio_returns(weights, returns)
        total_return = (1 + portfolio_returns).prod() - 1
        years = len(returns) / 252
        if years <= 0:
            return 0
        return ((1 + total_return) ** (1/years) - 1) * 100
    
    def equal_weights(self, n_assets: int) -> np.ndarray:
        # Simplest possible allocation — divide 1.0 evenly across all assets.
        # Used as the starting point (initial guess) for every optimizer in this class.
        return np.ones(n_assets) / n_assets
    
    def risk_parity(self, returns: pd.DataFrame, bounds: List[Tuple]) -> np.ndarray:
        # Risk parity means each asset contributes the same amount of risk to the portfolio,
        # rather than each asset having the same dollar weight.
        # The objective minimizes the squared difference between each asset's actual risk
        # contribution and the target (total portfolio vol / number of assets).
        # If the optimizer fails to converge we fall back to equal weights rather than crashing.
        cov_matrix = returns.cov() * 252
        n_assets = len(returns.columns)
        
        def portfolio_volatility(weights):
            return np.sqrt(weights @ cov_matrix @ weights)
        
        def risk_contributions(weights):
            portfolio_vol = portfolio_volatility(weights)
            if portfolio_vol == 0:
                return np.ones(n_assets) / n_assets
            marginal_risk = cov_matrix @ weights / portfolio_vol
            return weights * marginal_risk
        
        def objective(weights):
            rc = risk_contributions(weights)
            target_rc = portfolio_volatility(weights) / n_assets
            return np.sum((rc - target_rc) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(
            objective,
            self.equal_weights(n_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            # Fallback to equal weights if optimization fails
            return self.equal_weights(n_assets)
        
        return result.x
    
    def minimize_volatility(self, returns: pd.DataFrame, bounds: List[Tuple]) -> np.ndarray:
        # The objective is simply the portfolio's annualized standard deviation.
        # scipy.minimize will nudge the weights until it finds the combination that
        # produces the smallest possible volatility while keeping weights summed to 1.
        cov_matrix = returns.cov() * 252
        
        def objective(weights):
            return np.sqrt(weights @ cov_matrix @ weights)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(
            objective,
            self.equal_weights(len(returns.columns)),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            raise ValueError("Volatility minimization failed to converge")
        
        return result.x
    
    def minimize_drawdown(self, returns: pd.DataFrame, bounds: List[Tuple]) -> np.ndarray:
        # Drawdown is expensive to compute on every optimizer iteration since it requires
        # replaying the full return history, but it's the most direct way to limit losses.
        # We pass calculate_max_drawdown directly as the objective and let SLSQP do the work.
        
        def objective(weights):
            return self.calculate_max_drawdown(weights, returns)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(
            objective,
            self.equal_weights(len(returns.columns)),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            raise ValueError("Drawdown minimization failed to converge")
        
        return result.x
    
    def maximize_sharpe_ratio(self, returns: pd.DataFrame, bounds: List[Tuple]) -> np.ndarray:
        # scipy only minimizes, so we flip the sign of the Sharpe ratio to turn
        # "maximize Sharpe" into "minimize negative Sharpe" — same problem, same solution.
        
        def negative_sharpe(weights):
            return -self.calculate_portfolio_sharpe(weights, returns)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(
            negative_sharpe,
            self.equal_weights(len(returns.columns)),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            raise ValueError("Sharpe ratio maximization failed to converge")
        
        return result.x