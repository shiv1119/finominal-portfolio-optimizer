import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class WeightBounds:
    # Simple container to hold the lower and upper weight limits for one asset.
    # We use a dataclass here so we don't have to write __init__ manually.
    lower: float
    upper: float

class ConstraintValidator:
    
    def __init__(self):
        # Default bounds — by default any asset can range from 0% to 100%.
        # These get overridden per-asset if the caller passes in custom constraints.
        self.min_weight_default = 0.0
        self.max_weight_default = 1.0
    
    def create_bounds(
        self,
        n_assets: int,
        weight_constraints: Optional[Dict[str, Dict[str, float]]] = None,
        tickers: Optional[List[str]] = None
    ) -> List[Tuple[float, float]]:
        # Start by giving every asset the default (0.0, 1.0) bounds.
        # Then loop through the tickers and tighten the bounds for any asset
        # that has a custom min or max weight coming in from the API request.
        # Note: the API sends weights as percentages (e.g. 20), so we divide by 100
        # to convert to decimals (0.2) that the optimizer expects.
        bounds = [(self.min_weight_default, self.max_weight_default) for _ in range(n_assets)]
        
        if weight_constraints and tickers:
            for i, ticker in enumerate(tickers):
                if ticker in weight_constraints:
                    constraint = weight_constraints[ticker]
                    if 'min_weight' in constraint and constraint['min_weight'] is not None:
                        bounds[i] = (constraint['min_weight'] / 100, bounds[i][1])
                    if 'max_weight' in constraint and constraint['max_weight'] is not None:
                        bounds[i] = (bounds[i][0], constraint['max_weight'] / 100)
        
        return bounds
    
    def validate_portfolio_constraints(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        portfolio_constraints: Optional[Dict[str, float]],
        portfolio_service
    ) -> bool:
        # If no constraints were passed in, nothing to check — return True immediately.
        # Otherwise, run each constraint check one by one and bail out early with False
        # the moment any single constraint is violated. All constraints must pass to return True.
        if not portfolio_constraints:
            return True
        
        # Check CAGR constraint
        if 'min_cagr' in portfolio_constraints and portfolio_constraints['min_cagr'] is not None:
            cagr = portfolio_service.calculate_cagr(weights, returns)
            if cagr < portfolio_constraints['min_cagr']:
                return False
        
        # Check volatility lower bound — portfolio must be AT LEAST this volatile.
        # Useful when the user wants to avoid cash-like allocations that technically
        # minimize volatility but produce no meaningful return.
        if 'min_volatility' in portfolio_constraints and portfolio_constraints['min_volatility'] is not None:
            vol = portfolio_service.calculate_portfolio_volatility(weights, returns) * 100
            if vol < portfolio_constraints['min_volatility']:
                return False

        # Check volatility upper bound — portfolio must be NO MORE volatile than this.
        if 'max_volatility' in portfolio_constraints and portfolio_constraints['max_volatility'] is not None:
            vol = portfolio_service.calculate_portfolio_volatility(weights, returns) * 100
            if vol > portfolio_constraints['max_volatility']:
                return False
        
        # Check drawdown constraint
        if 'max_drawdown' in portfolio_constraints and portfolio_constraints['max_drawdown'] is not None:
            dd = portfolio_service.calculate_max_drawdown(weights, returns)
            if dd > portfolio_constraints['max_drawdown']:
                return False
        
        return True
    
    def validate_dividend_yield_constraint(
        self,
        weights: np.ndarray,
        tickers: List[str],
        dividend_yields: Dict[str, float],
        min_dividend_yield: Optional[float]
    ) -> bool:
        # Skip the check entirely if no minimum dividend yield was requested.
        # Otherwise, compute the weighted average dividend yield across all holdings
        # and check if it clears the minimum threshold.
        # If a ticker has no dividend data we treat its yield as 0 rather than crashing.
        if min_dividend_yield is None:
            return True
        
        portfolio_div_yield = 0.0
        for i, ticker in enumerate(tickers):
            div_yield = dividend_yields.get(ticker, 0.0)
            portfolio_div_yield += weights[i] * div_yield
        
        return portfolio_div_yield >= min_dividend_yield
    
    def build_scipy_portfolio_constraints(
        self,
        portfolio_constraints: Optional[Dict],
        returns: pd.DataFrame,
        tickers: List[str],
        dividend_yields: Dict[str, float],
        portfolio_optimizer
    ) -> List[Dict]:
        # Converts portfolio-level constraints into SciPy-compatible constraint dicts
        # so they are enforced DURING optimization, not just validated after.
        # Each entry follows the SciPy format: {"type": "ineq", "fun": callable}
        # where "ineq" means fun(weights) >= 0 must hold at the solution.
        # This is how we wire min_cagr, min/max_volatility, max_drawdown, and
        # min_dividend_yield directly into the SLSQP solver.
        scipy_constraints = []

        if not portfolio_constraints:
            return scipy_constraints

        # Min CAGR — portfolio annualised return must be at least this value.
        if portfolio_constraints.get('min_cagr') is not None:
            min_cagr = portfolio_constraints['min_cagr']
            scipy_constraints.append({
                'type': 'ineq',
                'fun': lambda w, r=returns, mc=min_cagr: (
                    portfolio_optimizer.calculate_cagr(w, r) - mc
                )
            })

        # Min volatility — lower bound of the volatility range.
        # fun >= 0  =>  actual_vol - min_vol >= 0  =>  actual_vol >= min_vol
        if portfolio_constraints.get('min_volatility') is not None:
            min_vol = portfolio_constraints['min_volatility'] / 100  # % → decimal
            scipy_constraints.append({
                'type': 'ineq',
                'fun': lambda w, r=returns, mv=min_vol: (
                    portfolio_optimizer.calculate_portfolio_volatility(w, r) - mv
                )
            })

        # Max volatility — upper bound of the volatility range.
        # fun >= 0  =>  max_vol - actual_vol >= 0  =>  actual_vol <= max_vol
        if portfolio_constraints.get('max_volatility') is not None:
            max_vol = portfolio_constraints['max_volatility'] / 100  # % → decimal
            scipy_constraints.append({
                'type': 'ineq',
                'fun': lambda w, r=returns, mv=max_vol: (
                    mv - portfolio_optimizer.calculate_portfolio_volatility(w, r)
                )
            })

        # Max drawdown — peak-to-trough loss must not exceed this threshold.
        # fun >= 0  =>  max_dd - actual_dd >= 0  =>  actual_dd <= max_dd
        if portfolio_constraints.get('max_drawdown') is not None:
            max_dd = portfolio_constraints['max_drawdown']
            scipy_constraints.append({
                'type': 'ineq',
                'fun': lambda w, r=returns, md=max_dd: (
                    md - portfolio_optimizer.calculate_max_drawdown(w, r)
                )
            })

        # Min dividend yield — weighted portfolio yield must be at least this value.
        # fun >= 0  =>  actual_yield - min_yield >= 0  =>  actual_yield >= min_yield
        if portfolio_constraints.get('min_dividend_yield') is not None:
            min_yield = portfolio_constraints['min_dividend_yield'] / 100  # % → decimal
            dy_array = np.array([dividend_yields.get(t, 0.0) for t in tickers])
            scipy_constraints.append({
                'type': 'ineq',
                'fun': lambda w, dy=dy_array, my=min_yield: float(np.dot(w, dy)) - my
            })

        return scipy_constraints

    def check_feasibility(
        self,
        bounds: List[Tuple[float, float]],
        n_assets: int
    ) -> bool:
        # Before running the optimizer, make sure a valid solution is even possible.
        # If the sum of all lower bounds already exceeds 1.0, or the sum of all upper
        # bounds can't reach 1.0, then no combination of weights will ever sum to 100% —
        # so we catch that early and return False instead of wasting time optimizing.
        min_possible_sum = sum(lower for lower, _ in bounds)
        max_possible_sum = sum(upper for _, upper in bounds)
        
        return min_possible_sum <= 1.0 <= max_possible_sum