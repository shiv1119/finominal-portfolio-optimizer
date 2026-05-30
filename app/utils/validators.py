import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError

class InputValidator:
    
    @staticmethod
    def validate_ticker(ticker: str, available_tickers: List[str]) -> bool:
        # Simple membership check — returns True only if the ticker exists in
        # the list of funds we actually have data for.
        return ticker in available_tickers
    
    @staticmethod
    def validate_weight_sum(weights: List[float], tolerance: float = 0.01) -> bool:
        # We allow a small tolerance of 0.01% because floating point arithmetic
        # can leave the sum at something like 99.99999 instead of exactly 100.
        total = sum(weights)
        return abs(total - 100) <= tolerance
    
    @staticmethod
    def validate_weight_range(weights: List[float]) -> bool:
        # Checks that no weight is negative — short selling is not supported,
        # so every weight must be zero or positive.
        return all(w >= 0 for w in weights)
    
    @staticmethod
    def validate_constraints(
        weight_constraints: Optional[Dict[str, Dict[str, float]]],
        tickers: List[str]
    ) -> Dict[str, List[str]]:
        # Walks through every constraint the caller provided and collects all
        # problems into an errors dict rather than stopping at the first one —
        # this way the caller gets all their mistakes back in a single response.
        # We check three things per ticker: the ticker exists in the request,
        # min/max are individually within 0–100, and min doesn't exceed max.
        errors = {"weight_constraints": [], "portfolio_constraints": []}
        
        if weight_constraints:
            for ticker, constraint in weight_constraints.items():
                if ticker not in tickers:
                    errors["weight_constraints"].append(f"Ticker '{ticker}' in constraints not found in securities")
                
                if 'min_weight' in constraint and constraint['min_weight'] is not None:
                    if constraint['min_weight'] < 0 or constraint['min_weight'] > 100:
                        errors["weight_constraints"].append(f"Invalid min_weight for {ticker}: must be between 0 and 100")
                
                if 'max_weight' in constraint and constraint['max_weight'] is not None:
                    if constraint['max_weight'] < 0 or constraint['max_weight'] > 100:
                        errors["weight_constraints"].append(f"Invalid max_weight for {ticker}: must be between 0 and 100")
                
                if ('min_weight' in constraint and 'max_weight' in constraint and 
                    constraint['min_weight'] is not None and constraint['max_weight'] is not None):
                    if constraint['min_weight'] > constraint['max_weight']:
                        errors["weight_constraints"].append(f"min_weight > max_weight for {ticker}")
        
        return errors
    
    @staticmethod
    def validate_strategy(strategy: str, available_strategies: List[str]) -> bool:
        # Checks the strategy string against the list of supported ones.
        # Keeping this as a separate method lets us swap in a database lookup later
        # without changing every caller.
        return strategy in available_strategies
    
    @staticmethod
    def validate_factor(factor: Optional[str], available_factors: List[str] = None) -> bool:
        # Returns True immediately if no factor was provided — it's only required
        # for the factor exposure strategy, so None is valid in all other cases.
        # We lowercase before comparing so "Momentum" and "momentum" both pass.
        if factor is None:
            return True
        available_factors = available_factors or ['momentum', 'value', 'size']
        return factor.lower() in available_factors

class DataValidator:
    
    @staticmethod
    def validate_returns_data(returns_df, min_required_days: int = 252) -> Dict[str, Any]:
        # Runs a series of quality checks on the returns DataFrame and collects
        # all issues into a list rather than raising on the first problem.
        # We check for: empty data, fewer days than the minimum (default 1 year / 252 trading days),
        # too many missing values (more than 5% is suspicious), and any security with zero variance
        # which would make the optimizer divide by zero.
        issues = []
        
        if returns_df is None or returns_df.empty:
            issues.append("Returns data is empty")
            return {"is_valid": False, "issues": issues, "warnings": []}
        
        # Check for sufficient data
        if len(returns_df) < min_required_days:
            issues.append(f"Insufficient data: only {len(returns_df)} days, need {min_required_days}")
        
        # Check for missing values
        missing_pct = returns_df.isnull().sum().sum() / (returns_df.shape[0] * returns_df.shape[1]) * 100
        if missing_pct > 5:
            issues.append(f"High percentage of missing values: {missing_pct:.2f}%")
        
        # Check for constant series
        for col in returns_df.columns:
            if returns_df[col].std() < 1e-6:
                issues.append(f"Constant returns for {col}")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": [],
            "n_days": len(returns_df),
            "n_assets": len(returns_df.columns),
            "missing_percentage": missing_pct
        }
    
    @staticmethod
    def validate_factor_data(factor_df, min_required_days: int = 126) -> Dict[str, Any]:
        # Similar to validate_returns_data but for factor data, which has a lower
        # minimum threshold (126 days ≈ half a year) since factor strategies
        # can work with less history than a full returns optimization.
        # We also check that all three expected factor columns are present.
        issues = []
        
        if factor_df is None or factor_df.empty:
            issues.append("Factor data is empty")
            return {"is_valid": False, "issues": issues, "warnings": []}
        
        # Check for required factors
        required_factors = ['momentum', 'value', 'size']
        missing_factors = [f for f in required_factors if f not in factor_df.columns]
        if missing_factors:
            issues.append(f"Missing factors: {missing_factors}")
        
        # Check for sufficient data
        if len(factor_df) < min_required_days:
            issues.append(f"Insufficient factor data: only {len(factor_df)} days")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": [],
            "n_days": len(factor_df),
            "available_factors": [c for c in required_factors if c in factor_df.columns]
        }

class RequestValidator:
    
    @staticmethod
    def validate_securities_unique(securities: List[Dict]) -> bool:
        # Compares the total count of tickers to the count of unique tickers.
        # If they differ, at least one ticker was submitted twice which would
        # cause the optimizer to double-count that position.
        tickers = [s.get('ticker') for s in securities]
        return len(tickers) == len(set(tickers))
    
    @staticmethod
    def validate_weights_not_extreme(weights: List[float], max_allowed: float = 100) -> bool:
        # Ensures every weight sits between 0 and 100 — catches cases where
        # someone accidentally passes raw decimal weights (e.g. 0.25) instead
        # of percentages (e.g. 25.0).
        return all(0 <= w <= max_allowed for w in weights)
    
    @staticmethod
    def validate_optimization_request(request_data: Dict) -> Dict[str, Any]:
        # Top-level validation that ties together all the individual checks above.
        # We accumulate every error into a dict so the caller sees all problems at once
        # rather than fixing one and resubmitting only to hit another.
        # Special-cased: factor exposure strategy requires factor_to_optimize to be set.
        errors = {}
        
        # Check securities
        if 'securities' not in request_data:
            errors['securities'] = "Missing securities field"
        elif not request_data['securities']:
            errors['securities'] = "Securities list cannot be empty"
        elif not RequestValidator.validate_securities_unique(request_data['securities']):
            errors['securities'] = "Duplicate tickers found in securities"
        
        # Check strategy
        if 'strategy' not in request_data:
            errors['strategy'] = "Missing strategy field"
        
        # Check factor exposure requirements
        if request_data.get('strategy') == 'optimize_factor_exposure':
            if 'factor_to_optimize' not in request_data:
                errors['factor_to_optimize'] = "factor_to_optimize required for factor exposure strategy"
            elif not InputValidator.validate_factor(request_data.get('factor_to_optimize')):
                errors['factor_to_optimize'] = "Invalid factor. Must be: momentum, value, size"
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }