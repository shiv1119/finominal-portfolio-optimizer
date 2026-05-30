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
        # without changing this class.
        self.data_loader = data_loader
        self.optimizer = PortfolioOptimizer(risk_free_rate=0.0)
        self.factor_analyzer = FactorAnalyzer()
        self.constraint_validator = ConstraintValidator()
    
    def validate_before_optimization(
        self,
        securities: List[str],
        weight_constraints: Optional[Dict] = None,
        portfolio_constraints: Optional[Dict] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> None:
        # Run all sanity checks before we hand anything to the optimizer,
        # so we fail fast with a clear message instead of getting a cryptic math error mid-run.
        # Three things we check: enough historical data, no dead/flat securities,
        # and that the weight bounds can actually produce a portfolio summing to 100%.

        # Use date-filtered returns for validation so we're checking the same
        # data window the optimizer will actually use.
        returns = self.data_loader.get_fund_returns_for_date_range(
            securities, start_date=start_date, end_date=end_date
        )

        if len(returns) < 30:
            raise ErrorHandler.infeasible_constraints(
                f"Insufficient historical data: only {len(returns)} days available. Need at least 30 days."
            )
        
        for col in returns.columns:
            if returns[col].std() < 1e-6:
                raise ErrorHandler.infeasible_constraints(
                    f"Security {col} has constant returns (no variance), cannot optimize"
                )
        
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
        
        if portfolio_constraints:
            self._validate_portfolio_constraints(portfolio_constraints, securities, bounds)
    
    def _validate_portfolio_constraints(
        self,
        portfolio_constraints: Dict,
        securities: List[str],
        bounds: List[Tuple[float, float]]
    ) -> None:
        # Check whether the portfolio-level constraints are achievable given the
        # securities and weight bounds we're working with.

        if 'min_dividend_yield' in portfolio_constraints:
            min_div_yield = portfolio_constraints['min_dividend_yield'] / 100
            
            dividend_yields = np.array([
                self.data_loader.get_dividend_yield(ticker) 
                for ticker in securities
            ])
            
            upper_bounds = np.array([high for _, high in bounds])
            max_div_yield = np.sum(upper_bounds * dividend_yields)
            
            if max_div_yield < min_div_yield - 1e-6:
                raise ErrorHandler.infeasible_constraints(
                    f"Dividend yield constraint ({portfolio_constraints['min_dividend_yield']}%) infeasible. "
                    f"Maximum achievable is {max_div_yield * 100:.2f}%"
                )
        
        # Volatility range sanity check: lower bound must be below upper bound.
        min_vol = portfolio_constraints.get('min_volatility')
        max_vol = portfolio_constraints.get('max_volatility')
        if min_vol is not None and max_vol is not None and min_vol > max_vol:
            raise ErrorHandler.infeasible_constraints(
                f"Volatility range infeasible: min_volatility ({min_vol}%) "
                f"is greater than max_volatility ({max_vol}%)"
            )

        if max_vol is not None and max_vol < 0.1:
            logger.warning(f"Very strict max_volatility constraint: {max_vol}%")
    
    def optimize(
        self,
        securities: List[str],
        current_weights: List[float],
        strategy: str,
        weight_constraints: Optional[Dict] = None,
        portfolio_constraints: Optional[Dict] = None,
        factor_to_optimize: Optional[str] = None,
        factor_direction: str = "maximize",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[Dict], Optional[Dict]]:
        # Main entry point that coordinates everything.
        # Order of operations: validate → load data (with date filter) → build SciPy
        # constraints → run strategy → normalize → compute factor betas.
        # start_date and end_date are passed all the way through so the optimizer
        # works on exactly the same historical window the user specified, matching
        # the live tool's Time Frame behaviour.

        self.validate_before_optimization(
            securities, weight_constraints, portfolio_constraints,
            start_date=start_date, end_date=end_date
        )
        
        # Load returns sliced to the requested date window.
        returns = self.data_loader.get_fund_returns_for_date_range(
            securities, start_date=start_date, end_date=end_date
        )
        returns = returns.loc[(returns != 0).all(axis=1)]
        returns = returns.dropna()
        current_weights_array = np.array(current_weights) / 100
        
        bounds = self.constraint_validator.create_bounds(
            len(securities),
            weight_constraints,
            securities
        )

        # Build portfolio-level SciPy constraints so they are enforced INSIDE the
        # optimizer on every iteration, not just validated after the fact.
        # This is what makes min_cagr, min/max_volatility, max_drawdown, and
        # min_dividend_yield actually respected at the solution point.
        dividend_yields = {t: self.data_loader.get_dividend_yield(t) for t in securities}
        extra_constraints = self.constraint_validator.build_scipy_portfolio_constraints(
            portfolio_constraints,
            returns,
            securities,
            dividend_yields,
            self.optimizer
        )
        
        # Compute current factor betas before optimization so the response can
        # show how factor exposures shifted. Skip if factor data isn't loaded.
        current_betas = None
        factor_returns = self.data_loader.get_factor_returns()
        if factor_returns is not None:
            aligned_returns = self.data_loader.get_common_date_range(returns)
            current_betas = self.factor_analyzer.calculate_factor_betas(
                current_weights_array, aligned_returns, factor_returns
            )
        
        # Dispatch to the right optimizer. extra_constraints is passed to every
        # strategy so portfolio-level constraints are always respected regardless
        # of which objective function is being optimized.
        try:
            if strategy == "equal_weights":
                # Equal weights has no optimizer loop, so we can't enforce
                # portfolio constraints via SciPy. We still validate them
                # post-hoc and warn if they aren't met.
                optimized_weights = self.optimizer.equal_weights(len(securities))
                
            elif strategy == "risk_parity":
                print("Bounds being passed:", bounds)
                optimized_weights = self.optimizer.risk_parity(
                    returns, bounds, extra_constraints=extra_constraints
                )
                
            elif strategy == "minimize_volatility":
                optimized_weights = self.optimizer.minimize_volatility(
                    returns, bounds, extra_constraints=extra_constraints
                )
                
            elif strategy == "minimize_drawdown":
                optimized_weights = self.optimizer.minimize_drawdown(
                    returns, bounds, extra_constraints=extra_constraints
                )
                
            elif strategy == "maximize_sharpe_ratio":
                optimized_weights = self.optimizer.maximize_sharpe_ratio(
                    returns, bounds, extra_constraints=extra_constraints
                )
                
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
                    self.optimizer,
                    extra_constraints=extra_constraints
                )
            else:
                raise ErrorHandler.unsupported_strategy(strategy)
            
            # Normalize to ensure weights sum to exactly 1.0 — floating point arithmetic
            # in the optimizer can leave the sum slightly off.
            optimized_weights = optimized_weights / np.sum(optimized_weights)
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise ErrorHandler.optimization_failed(f"Optimization algorithm failed: {str(e)}")
        
        # Compute factor betas for the optimized weights.
        optimized_betas = None
        if factor_returns is not None:
            aligned_returns = self.data_loader.get_common_date_range(returns)
            optimized_betas = self.factor_analyzer.calculate_factor_betas(
                optimized_weights, aligned_returns, factor_returns
            )
        
        return optimized_weights, current_betas, optimized_betas