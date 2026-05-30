from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum

class OptimizationStrategy(str, Enum):
    # Enum listing all supported optimization strategies.
    # Using an Enum instead of plain strings prevents typos and makes it easy to see
    # every valid option in one place.
    EQUAL_WEIGHTS = "equal_weights"
    RISK_PARITY = "risk_parity"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MINIMIZE_VOLATILITY = "minimize_volatility"
    MAXIMIZE_SHARPE_RATIO = "maximize_sharpe_ratio"
    OPTIMIZE_FACTOR_EXPOSURE = "optimize_factor_exposure"

class WeightConstraint(BaseModel):
    # Defines the min/max weight boundaries for a single security.
    # Both fields are optional — only set the ones you care about.
    # ge=0 and le=100 enforce that weights are valid percentages (0–100).
    min_weight: Optional[float] = Field(None, ge=0, le=100, description="Minimum weight percentage")
    max_weight: Optional[float] = Field(None, ge=0, le=100, description="Maximum weight percentage")
    
    @validator('min_weight', 'max_weight')
    def validate_weight_range(cls, v, values, **kwargs):
        # Redundant safety check — raises a clear error if a weight somehow slips outside 0–100.
        if v is not None and (v < 0 or v > 100):
            raise ValueError(f"Weight must be between 0 and 100, got {v}")
        return v
    
    @validator('max_weight')
    def validate_min_max(cls, v, values, **kwargs):
        # Ensures min_weight never exceeds max_weight (e.g. min=60, max=40 would be rejected).
        if 'min_weight' in values and values['min_weight'] is not None and v is not None:
            if values['min_weight'] > v:
                raise ValueError(f"min_weight ({values['min_weight']}) cannot be greater than max_weight ({v})")
        return v

class PortfolioConstraints(BaseModel):
    # Holds portfolio-level performance constraints that the optimizer must respect.
    # All fields are optional — only include the metrics you want to enforce.
    min_cagr: Optional[float] = Field(None, description="Minimum CAGR requirement")
    max_volatility: Optional[float] = Field(None, ge=0, description="Maximum volatility constraint")
    max_drawdown: Optional[float] = Field(None, ge=0, le=100, description="Maximum drawdown constraint")
    min_dividend_yield: Optional[float] = Field(None, ge=0, description="Minimum dividend yield requirement")

class SecurityInput(BaseModel):
    # Represents a single security in the input list, identified by ticker and its
    # current percentage allocation in the portfolio.
    ticker: str = Field(..., description="Security ticker symbol")
    current_weight: float = Field(..., ge=0, le=100, description="Current weight percentage")

class OptimizationRequest(BaseModel):
    # The main request body sent to the optimization API.
    # It carries the list of securities, the chosen strategy, and any optional constraints.
    securities: List[SecurityInput] = Field(..., min_items=1, description="List of securities with current weights")
    strategy: OptimizationStrategy = Field(..., description="Optimization strategy to apply")
    weight_constraints: Optional[Dict[str, WeightConstraint]] = Field(None, description="Per-security weight constraints")
    portfolio_constraints: Optional[PortfolioConstraints] = Field(None, description="Portfolio-level constraints")
    factor_to_optimize: Optional[str] = Field(None, description="Factor to optimize (momentum/value/size) for factor exposure strategy")
    factor_direction: Optional[str] = Field("maximize", description="maximize or minimize factor exposure")
    
    @validator('securities')
    def validate_weights_sum(cls, v):
        # Rejects the request early if the security weights don't add up to exactly 100%,
        # since a portfolio that doesn't sum to 100% is mathematically invalid.
        total_weight = sum(sec.current_weight for sec in v)
        if abs(total_weight - 100) > 0.01:  # Allow 0.01% floating point tolerance
            raise ValueError(f"Current weights must sum to 100%, got {total_weight}%")
        return v
    
    @validator('factor_to_optimize')
    def validate_factor(cls, v, values):
        # When the strategy is OPTIMIZE_FACTOR_EXPOSURE, makes sure factor_to_optimize
        # is provided and is one of the three supported values.
        if values.get('strategy') == OptimizationStrategy.OPTIMIZE_FACTOR_EXPOSURE:
            if v is None:
                raise ValueError("factor_to_optimize is required for factor exposure optimization")
            if v.lower() not in ['momentum', 'value', 'size']:
                raise ValueError("factor_to_optimize must be one of: momentum, value, size")
        return v.lower() if v else v

class AllocationChange(BaseModel):
    # Captures the before/after weight for one security so the caller can see
    # exactly how much each position changed after optimization.
    ticker: str
    security_name: str
    current_weight: float
    optimized_weight: float
    change: float

class FactorBetas(BaseModel):
    # Stores the three factor betas (value, momentum, size) for a portfolio.
    # Each beta measures how much the portfolio is exposed to that particular factor.
    value: float
    momentum: float
    size: float

class FactorBetasResponse(BaseModel):
    # Pairs the factor betas of the original and optimized portfolios side by side
    # so the caller can compare how factor exposures shifted after optimization.
    current_portfolio: FactorBetas
    optimized_portfolio: FactorBetas

class OptimizationResponse(BaseModel):
    # The response returned by the API after running an optimization.
    # Contains the strategy used, per-security allocation changes, and optionally
    # the factor betas if a factor exposure strategy was selected.
    optimization_strategy: str
    allocation_changes: List[AllocationChange]
    factor_betas: Optional[FactorBetasResponse] = None

class HealthResponse(BaseModel):
    # A lightweight health-check response used to confirm the service is running
    # and to surface what funds, strategies, and date ranges are currently available.
    status: str
    available_funds: List[str]
    available_strategies: List[str]
    data_date_range: Dict[str, str]