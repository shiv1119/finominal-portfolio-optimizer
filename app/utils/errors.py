from enum import Enum
from typing import Dict, Any, Optional
from fastapi import HTTPException, status

class ErrorCode(str, Enum):
    # Enum of all error codes the API can return, grouped by HTTP status category.
    # Having them in one place means we never typo an error string and every part
    # of the codebase speaks the same language when something goes wrong.
    
    # Validation Errors (400)
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_TICKER = "INVALID_TICKER"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    WEIGHTS_NOT_SUM_TO_100 = "WEIGHTS_NOT_SUM_TO_100"
    INVALID_WEIGHT_CONSTRAINT = "INVALID_WEIGHT_CONSTRAINT"
    INFEASIBLE_CONSTRAINTS = "INFEASIBLE_CONSTRAINTS"
    UNSUPPORTED_STRATEGY = "UNSUPPORTED_STRATEGY"
    
    # Data Errors (404)
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    FUND_NOT_FOUND = "FUND_NOT_FOUND"
    
    # Optimization Errors (422)
    OPTIMIZATION_FAILED = "OPTIMIZATION_FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    
    # Server Errors (500)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    DATA_LOAD_ERROR = "DATA_LOAD_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"


class PortfolioOptimizerException(HTTPException):
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Dict[str, Any] = None
    ):
        # Extends FastAPI's HTTPException so our custom errors flow through
        # FastAPI's error handling pipeline automatically without extra wiring.
        # We package the error_code, human-readable message, and any extra details
        # into the `detail` field so the API response is always structured the same way.
        self.error_code = error_code
        self.details = details or {}
        super().__init__(
            status_code=status_code,
            detail={
                "error_code": error_code,
                "message": message,
                "details": self.details
            }
        )


class ErrorHandler:
    
    @staticmethod
    def invalid_request(message: str, details: Dict[str, Any] = None):
        # Generic 400 for when the request body is malformed or fails business logic
        # that Pydantic validation alone doesn't cover.
        return PortfolioOptimizerException(
            ErrorCode.INVALID_REQUEST,
            message,
            status.HTTP_400_BAD_REQUEST,
            details
        )
    
    @staticmethod
    def invalid_ticker(ticker: str):
        # 404 raised when a ticker the caller sent doesn't exist in our loaded data.
        # We include the ticker in details so the client knows exactly which one failed.
        return PortfolioOptimizerException(
            ErrorCode.INVALID_TICKER,
            f"Ticker '{ticker}' not found in available funds",
            status.HTTP_404_NOT_FOUND,
            {"ticker": ticker}
        )
    
    @staticmethod
    def missing_field(field_name: str):
        # 400 for when a required field is absent from the request.
        # Pydantic catches most of these, but this covers runtime checks
        # where a field is conditionally required based on another field's value.
        return PortfolioOptimizerException(
            ErrorCode.MISSING_REQUIRED_FIELD,
            f"Required field '{field_name}' is missing",
            status.HTTP_400_BAD_REQUEST,
            {"field": field_name}
        )
    
    @staticmethod
    def weights_not_sum_to_100(current_sum: float):
        # 400 raised when the caller's security weights add up to something other than 100%.
        # We echo back what the sum actually was so they can fix it without guessing.
        return PortfolioOptimizerException(
            ErrorCode.WEIGHTS_NOT_SUM_TO_100,
            f"Current weights must sum to 100%, got {current_sum}%",
            status.HTTP_400_BAD_REQUEST,
            {"current_sum": current_sum, "expected_sum": 100}
        )
    
    @staticmethod
    def infeasible_constraints(message: str, details: Dict[str, Any] = None):
        # 422 used when the constraints are valid individually but impossible to satisfy
        # together — e.g. min weights that already exceed 100%, or a dividend yield
        # target higher than any combination of the given securities can achieve.
        return PortfolioOptimizerException(
            ErrorCode.INFEASIBLE_CONSTRAINTS,
            message,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            details
        )
    
    @staticmethod
    def unsupported_strategy(strategy: str):
        # 400 raised when the strategy string doesn't match any of our supported options.
        # We include the full list of valid strategies in the response so the caller
        # doesn't have to dig through docs to find out what's allowed.
        return PortfolioOptimizerException(
            ErrorCode.UNSUPPORTED_STRATEGY,
            f"Strategy '{strategy}' is not supported",
            status.HTTP_400_BAD_REQUEST,
            {
                "strategy": strategy, 
                "supported_strategies": [
                    "equal_weights", 
                    "risk_parity", 
                    "minimize_drawdown",
                    "minimize_volatility", 
                    "maximize_sharpe_ratio", 
                    "optimize_factor_exposure"
                ]
            }
        )
    
    @staticmethod
    def data_load_error(message: str):
        # 500 for when the Excel file fails to load at startup or during a request.
        # This is a server-side problem, not the caller's fault, hence the 500 status.
        return PortfolioOptimizerException(
            ErrorCode.DATA_LOAD_ERROR,
            f"Failed to load data: {message}",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    @staticmethod
    def optimization_failed(message: str, details: Dict[str, Any] = None):
        # 422 for when the optimizer runs but doesn't converge to a valid solution.
        # The request was understood and the data was fine — the math just couldn't
        # find a feasible answer, which is why we use 422 rather than 400 or 500.
        return PortfolioOptimizerException(
            ErrorCode.OPTIMIZATION_FAILED,
            f"Optimization failed: {message}",
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            details
        )
    
    @staticmethod
    def insufficient_data(message: str, details: Dict[str, Any] = None):
        # 422 for when the historical data exists but there's not enough of it
        # to produce a statistically meaningful result (e.g. fewer than 30 days).
        return PortfolioOptimizerException(
            ErrorCode.INSUFFICIENT_DATA,
            f"Insufficient data: {message}",
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            details
        )
    
    @staticmethod
    def internal_error(message: str, details: Dict[str, Any] = None):
        # Catch-all 500 for unexpected failures that don't fit a more specific category.
        return PortfolioOptimizerException(
            ErrorCode.INTERNAL_ERROR,
            message,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            details
        )
    
    @staticmethod
    def network_error(message: str = "Network error occurred"):
        # 503 for when a downstream network call fails — signals the service is temporarily
        # unavailable rather than broken, so clients know retrying may help.
        return PortfolioOptimizerException(
            ErrorCode.NETWORK_ERROR,
            message,
            status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    @staticmethod
    def fund_not_found(ticker: str):
        # 404 specifically for fund lookups — similar to invalid_ticker but used
        # when searching by fund identity rather than validating an input ticker list.
        return PortfolioOptimizerException(
            ErrorCode.FUND_NOT_FOUND,
            f"Fund with ticker '{ticker}' not found",
            status.HTTP_404_NOT_FOUND,
            {"ticker": ticker}
        )
    
    @staticmethod
    def data_not_found(data_type: str, identifier: str = None):
        # Generic 404 for any missing data resource beyond just funds or tickers.
        # The identifier is optional — if provided it gets appended to the message
        # so the caller knows exactly which record couldn't be found.
        message = f"{data_type} not found"
        if identifier:
            message += f": {identifier}"
        return PortfolioOptimizerException(
            ErrorCode.DATA_NOT_FOUND,
            message,
            status.HTTP_404_NOT_FOUND,
            {"data_type": data_type, "identifier": identifier}
        )
    
    @staticmethod
    def invalid_weight_constraint(message: str, details: Dict[str, Any] = None):
        # 400 for weight constraint values that are structurally invalid —
        # e.g. a min weight greater than the max weight for the same ticker.
        return PortfolioOptimizerException(
            ErrorCode.INVALID_WEIGHT_CONSTRAINT,
            message,
            status.HTTP_400_BAD_REQUEST,
            details
        )