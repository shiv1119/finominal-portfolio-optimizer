import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from app.services.data_loader import DataLoader
from app.core.optimizer import PortfolioOptimizer
from app.core.factors import FactorAnalyzer
from app.core.constraints import ConstraintValidator
from app.utils.errors import ErrorHandler
import logging

logger = logging.getLogger(__name__)

class PortfolioService:
    
    def __init__(self, data_loader: DataLoader):
        # Wire up all the components this service depends on.
        # We inject DataLoader from outside so it can be swapped or mocked in tests
        # without changing this class. The rest are internal helpers we always need.
        self.data_loader = data_loader
        self.optimizer = PortfolioOptimizer(risk_free_rate=0.0)
        self.factor_analyzer = FactorAnalyzer()
        self.constraint_validator = ConstraintValidator()
    
    def validate_before_optimization(
        self,
        securities: List[str],
        weight_constraints: Optional[Dict] = None,
        portfolio_constraints: Optional[Dict] = None
    ) -> None:
        # Run all sanity checks before we hand anything to the optimizer,
        # so we fail fast with a clear message instead of getting a cryptic math error mid-run.
        # Three things we check: enough historical data, no dead/flat securities,
        # and that the weight bounds can actually produce a portfolio summing to 100%.
        
        # Check if we have enough data
        returns = self.data_loader.get_fund_returns(securities)
        if len(returns) < 30:
            raise ErrorHandler.infeasible_constraints(
                f"Insufficient historical data: only {len(returns)} days available. Need at least 30 days."
            )
        
        # Check for constant returns (no variance)
        for col in returns.columns:
            if returns[col].std() < 1e-6:
                raise ErrorHandler.infeasible_constraints(
                    f"Security {col} has constant returns (no variance), cannot optimize"
                )
        
        # Validate weight constraints feasibility
        bounds = self.constraint_validator.create_bounds(
            len(securities),
            weight_constraints,
            securities
        )
        
        if not self.constraint_validator.check_feasibility(bounds, len(securities)):
            min_sum = sum(low for low, _ in bounds) * 100
            max_sum = sum(high for _, high in bounds) * 100
            raise ErrorHandler.infeasible_constraints(
                f"Weight constraints infeasible. With given min/max, "
                f"weights can only sum between {min_sum:.1f}% and {max_sum:.1f}%"
            )
        
        # Validate portfolio constraints
        if portfolio_constraints:
            self._validate_portfolio_constraints(portfolio_constraints, securities, bounds)
    
    def _validate_portfolio_constraints(
        self,
        portfolio_constraints: Dict,
        securities: List[str],
        bounds: List[Tuple[float, float]]
    ) -> None:
        # Check whether the portfolio-level constraints are even achievable given the
        # securities and weight bounds we're working with — better to catch this now
        # than let the optimizer spin for 1000 iterations and still fail.
        
        # For dividend yield we can do a hard check: multiply each asset's max allowed weight
        # by its dividend yield and sum them up — if that ceiling is still below the minimum
        # the caller asked for, no solution exists and we tell them immediately.
        # For volatility we can only really validate during the run, but we log a warning
        # if the threshold looks unrealistically tight (below 0.1% annual).
        if 'min_dividend_yield' in portfolio_constraints:
            min_div_yield = portfolio_constraints['min_dividend_yield'] / 100  # Convert to decimal
            
            dividend_yields = np.array([
                self.data_loader.get_dividend_yield(ticker) 
                for ticker in securities
            ])
            
            # Calculate maximum achievable dividend yield given upper bounds
            upper_bounds = np.array([high for _, high in bounds])
            max_div_yield = np.sum(upper_bounds * dividend_yields)
            
            if max_div_yield < min_div_yield - 1e-6:
                raise ErrorHandler.infeasible_constraints(
                    f"Dividend yield constraint ({portfolio_constraints['min_dividend_yield']}%) infeasible. "
                    f"Maximum achievable is {max_div_yield * 100:.2f}%"
                )
        
        # Volatility constraint - we can only check during optimization
        # but we can warn if it seems too strict
        if 'max_volatility' in portfolio_constraints:
            max_vol = portfolio_constraints['max_volatility']
            if max_vol < 0.1:  # Less than 0.1% annual volatility is unrealistic
                logger.warning(f"Very strict volatility constraint: {max_vol}%")
    
    def optimize(
        self,
        securities: List[str],
        current_weights: List[float],
        strategy: str,
        weight_constraints: Optional[Dict] = None,
        portfolio_constraints: Optional[Dict] = None,
        factor_to_optimize: Optional[str] = None,
        factor_direction: str = "maximize"
    ) -> Tuple[np.ndarray, Optional[Dict], Optional[Dict]]:
        # This is the main entry point that coordinates everything.
        # Order of operations: validate first, then load data, then run the chosen strategy,
        # then normalize weights, then compute factor betas for both before and after.
        # We capture current_betas before optimization so the response can show the diff.
        
        # Run pre-optimization validation
        self.validate_before_optimization(securities, weight_constraints, portfolio_constraints)
        
        # Get returns data
        returns = self.data_loader.get_fund_returns(securities)
        current_weights_array = np.array(current_weights) / 100
        
        # Create bounds
        bounds = self.constraint_validator.create_bounds(
            len(securities),
            weight_constraints,
            securities
        )
        
        # Calculate current factor betas before we change anything, so the response
        # can show how factor exposures shifted. Skip if factor data isn't loaded.
        current_betas = None
        factor_returns = self.data_loader.get_factor_returns()
        if factor_returns is not None:
            aligned_returns = self.data_loader.get_common_date_range(returns)
            current_betas = self.factor_analyzer.calculate_factor_betas(
                current_weights_array, aligned_returns, factor_returns
            )
        
        # Dispatch to the right optimizer based on the strategy string.
        # Each branch delegates to a focused method rather than putting all the math here.
        # Any exception from the optimizer gets caught and re-raised as a clean API error.
        try:
            if strategy == "equal_weights":
                optimized_weights = self.optimizer.equal_weights(len(securities))
                
            elif strategy == "risk_parity":
                optimized_weights = self.optimizer.risk_parity(returns, bounds)
                
            elif strategy == "minimize_volatility":
                optimized_weights = self.optimizer.minimize_volatility(returns, bounds)
                
            elif strategy == "minimize_drawdown":
                optimized_weights = self.optimizer.minimize_drawdown(returns, bounds)
                
            elif strategy == "maximize_sharpe_ratio":
                optimized_weights = self.optimizer.maximize_sharpe_ratio(returns, bounds)
                
            elif strategy == "optimize_factor_exposure":
                if factor_returns is None:
                    raise ErrorHandler.infeasible_constraints("Factor return data not available")
                
                aligned_returns = self.data_loader.get_common_date_range(returns)
                optimized_weights = self.factor_analyzer.optimize_factor_exposure(
                    aligned_returns,
                    factor_returns,
                    factor_to_optimize,
                    factor_direction,
                    bounds,
                    self.optimizer
                )
            else:
                raise ErrorHandler.unsupported_strategy(strategy)
            
            # Normalize to ensure weights sum to exactly 1.0 — floating point arithmetic
            # in the optimizer can leave the sum slightly off (e.g. 0.9999999 or 1.0000001).
            optimized_weights = optimized_weights / np.sum(optimized_weights)
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise ErrorHandler.optimization_failed(f"Optimization algorithm failed: {str(e)}")
        
        # Compute factor betas for the optimized weights so the caller can compare
        # before vs after. Same None-guard as above — skip if factor data isn't available.
        optimized_betas = None
        if factor_returns is not None:
            aligned_returns = self.data_loader.get_common_date_range(returns)
            optimized_betas = self.factor_analyzer.calculate_factor_betas(
                optimized_weights, aligned_returns, factor_returns
            )
        
        return optimized_weights, current_betas, optimized_betas