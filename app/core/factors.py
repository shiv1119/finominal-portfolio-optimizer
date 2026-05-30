import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.stats import linregress
from typing import List

class FactorAnalyzer:
    
    def __init__(self):
        # The three factors we support — every analysis in this class revolves around these.
        # Keeping them in a list makes it easy to loop over them consistently everywhere.
        self.factors = ['momentum', 'value', 'size']
    
    def calculate_factor_betas(
        self, 
        weights: np.ndarray, 
        returns: pd.DataFrame, 
        factor_returns: pd.DataFrame
    ) -> Dict[str, float]:
        # First we compute the portfolio's daily returns by dot-multiplying asset returns
        # with their weights, then align dates between the portfolio and factor return series
        # since they may not always cover the exact same date range.
        # We need at least 30 overlapping data points for the regression to be meaningful —
        # any fewer and we just return zeros rather than produce garbage betas.
        portfolio_returns = returns @ weights
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(factor_returns.index)
        if len(common_dates) < 30:
            return {factor: 0.0 for factor in self.factors}
        
        y = portfolio_returns[common_dates].values
        
        # Prepare factor matrix
        X = factor_returns.loc[common_dates, self.factors].values
        
        # We prepend a column of ones so that the regression also solves for alpha (intercept),
        # giving us: portfolio_return = alpha + beta_momentum*M + beta_value*V + beta_size*S
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # Try the full multiple regression first using least squares.
        # coefficients[0] is alpha, coefficients[1..3] are the three factor betas.
        # If that fails for any reason (singular matrix, bad data etc.) we fall back to
        # running a simple one-factor regression separately for each factor.
        try:
            # Using least squares
            coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            betas = {
                'momentum': float(coefficients[1]) if len(coefficients) > 1 else 0.0,
                'value': float(coefficients[2]) if len(coefficients) > 2 else 0.0,
                'size': float(coefficients[3]) if len(coefficients) > 3 else 0.0
            }
        except:
            # Fallback to simple linear regression per factor
            betas = {}
            for i, factor in enumerate(self.factors):
                X_single = factor_returns.loc[common_dates, factor].values
                slope, intercept, r_value, p_value, std_err = linregress(X_single, y)
                betas[factor] = float(slope)
        
        return betas
    
    def calculate_factor_exposure(
        self,
        weights: np.ndarray,
        fund_factor_loadings: pd.DataFrame
    ) -> Dict[str, float]:
        # Alternative to calculate_factor_betas for when we don't have factor return time-series.
        # Instead we use pre-computed per-fund factor loadings and take the weighted average
        # across all holdings to get the portfolio-level exposure for each factor.
        # If no loadings data was provided at all, we return zeros rather than crash.
        if fund_factor_loadings is None:
            return {factor: 0.0 for factor in self.factors}
        
        exposure = {}
        for factor in self.factors:
            if factor in fund_factor_loadings.columns:
                exposure[factor] = float(weights @ fund_factor_loadings[factor].values)
            else:
                exposure[factor] = 0.0
        
        return exposure
    
    def optimize_factor_exposure(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        factor_to_optimize: str,
        direction: str,
        bounds: List[Tuple],
        base_optimizer
    ) -> np.ndarray:
        # The objective function computes factor betas for a given set of weights and
        # returns the negative exposure when maximizing (because scipy.minimize always minimizes,
        # so flipping the sign turns a maximization into a minimization problem).
        from scipy.optimize import minimize
        
        def objective(weights):
            betas = self.calculate_factor_betas(weights, returns, factor_returns)
            exposure = betas.get(factor_to_optimize, 0)
            return -exposure if direction == "maximize" else exposure
        
        # We always enforce that weights sum to 1 (fully invested portfolio).
        # We also add a volatility cap of 30% to stop the optimizer from landing on
        # crazy concentrated portfolios that technically maximize the factor but are unusable.
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            # Add volatility constraint to avoid extreme portfolios
            {'type': 'ineq', 'fun': lambda x: 0.3 - base_optimizer.calculate_portfolio_volatility(x, returns)}
        ]
        
        result = minimize(
            objective,
            base_optimizer.equal_weights(len(returns.columns)),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        # If the optimizer failed with the volatility constraint, retry without it —
        # some edge cases (e.g. very few assets) make that constraint infeasible.
        # If it still fails after the retry, we raise an error rather than silently
        # return a bad result that would mislead the caller.
        if not result.success:
            # Try without volatility constraint
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            result = minimize(
                objective,
                base_optimizer.equal_weights(len(returns.columns)),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if not result.success:
                raise ValueError(f"Factor exposure optimization for {factor_to_optimize} failed")
        
        return result.x