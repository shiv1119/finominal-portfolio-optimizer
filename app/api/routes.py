from fastapi import APIRouter, HTTPException, status
from typing import List, Dict
import numpy as np
from app.api.models import (
    OptimizationRequest,
    OptimizationResponse,
    AllocationChange,
    FactorBetas,
    FactorBetasResponse,
    HealthResponse
)
from app.services.data_loader import DataLoader
from app.services.portfolio_service import PortfolioService
from app.utils.errors import ErrorHandler, ErrorCode, PortfolioOptimizerException
from app.utils.validators import RequestValidator, InputValidator, DataValidator
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Services are initialized once at module load time rather than on every request —
# loading the Excel file on every call would be way too slow.
# If initialization fails we raise immediately so the app won't start in a broken state.
try:
    data_loader = DataLoader(data_path="data", filename="data.xlsx")
    portfolio_service = PortfolioService(data_loader)
    logger.info("Services initialized successfully")
    logger.info(f"Available funds: {data_loader.get_all_funds()}")
except Exception as e:
    logger.error(f"Failed to initialize services: {str(e)}")
    raise

@router.get("/health", response_model=HealthResponse)
async def health_check():
    # Quick liveness check — confirms the service started correctly and returns
    # the list of available funds and strategies so clients can discover them
    # without hitting the full optimization endpoint.
    try:
        available_funds = data_loader.get_all_funds()
        return HealthResponse(
            status="healthy",
            available_funds=available_funds,
            available_strategies=["equal_weights", "risk_parity", "minimize_drawdown", 
                                  "minimize_volatility", "maximize_sharpe_ratio", 
                                  "optimize_factor_exposure"],
            data_date_range={
                "start": "Available",
                "end": "Available"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise ErrorHandler.internal_error("Health check failed")

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(request: OptimizationRequest):
    # Main endpoint — orchestrates the full request lifecycle in six clear steps
    # so it's easy to follow where a failure happened if something goes wrong.
    # Steps 1–4 are all about validating and preparing the data before we touch
    # the optimizer, because catching bad input early gives cleaner error messages.
    try:
        logger.info(f"Received optimization request with strategy: {request.strategy}")
        
        # 1 - Fist extract securities data
        # Pull tickers and weights out of the Pydantic models into plain lists
        # so the service layer doesn't need to know about request model structure.
        securities = [sec.ticker for sec in request.securities]
        current_weights = [sec.current_weight for sec in request.securities]
        
        logger.info(f"Securities: {securities}")
        logger.info(f"Current weights: {current_weights}")
        
        # 2 - Validate Tickers
        # Check every ticker against what we actually have data for —
        # a single unknown ticker would cause a silent KeyError deep in the optimizer.
        available_funds = data_loader.get_all_funds()
        logger.info(f"Available funds in data: {available_funds}")
        
        for ticker in securities:
            if ticker not in available_funds:
                raise ErrorHandler.invalid_ticker(ticker)
        
        # 3 -Now validate current weights
        # Even though Pydantic already checks this, we re-verify here as a belt-and-suspenders
        # guard in case the request was constructed programmatically and bypassed the model.
        total_weight = sum(current_weights)
        if abs(total_weight - 100) > 0.01:
            raise ErrorHandler.weights_not_sum_to_100(total_weight)
        
        # 4 - now prepare constraints
        # Convert Pydantic constraint models into plain dicts that the service layer
        # can work with — keeps the service layer decoupled from API model types.
        weight_constraints_dict = None
        if request.weight_constraints:
            weight_constraints_dict = {
                ticker: {
                    'min_weight': constraint.min_weight,
                    'max_weight': constraint.max_weight
                }
                for ticker, constraint in request.weight_constraints.items()
            }
        
        portfolio_constraints = None
        if request.portfolio_constraints:
            portfolio_constraints = request.portfolio_constraints.dict(exclude_none=True)
        
        # 5 - Now run optimization
        # Hand everything off to the service layer and get back the optimized weights
        # plus factor betas for both the original and optimized portfolios.
        optimized_weights, current_betas, optimized_betas = portfolio_service.optimize(
            securities=securities,
            current_weights=current_weights,
            strategy=request.strategy.value,
            weight_constraints=weight_constraints_dict,
            portfolio_constraints=portfolio_constraints,
            factor_to_optimize=request.factor_to_optimize,
            factor_direction=request.factor_direction
        )
        
        # 6 - Now prepare response
        # Convert raw numpy weights back to rounded percentages and pair each one
        # with its fund name and the change vs the original weight.
        # Factor betas are only included in the response when they were actually computed —
        # i.e. when factor return data was loaded and the strategy needed them.
        allocation_changes = []
        for i, security in enumerate(request.securities):
            fund_info = data_loader.get_fund_info(security.ticker)
            optimized_weight = round(optimized_weights[i] * 100, 2)
            
            allocation_changes.append(
                AllocationChange(
                    ticker=security.ticker,
                    security_name=fund_info.get('fund_name', security.ticker),
                    current_weight=security.current_weight,
                    optimized_weight=optimized_weight,
                    change=round(optimized_weight - security.current_weight, 2)
                )
            )
        
        # Now prepare factor betas response if available
        factor_betas_response = None
        if current_betas and optimized_betas:
            factor_betas_response = FactorBetasResponse(
                current_portfolio=FactorBetas(
                    value=round(current_betas.get('value', 0), 4),
                    momentum=round(current_betas.get('momentum', 0), 4),
                    size=round(current_betas.get('size', 0), 4)
                ),
                optimized_portfolio=FactorBetas(
                    value=round(optimized_betas.get('value', 0), 4),
                    momentum=round(optimized_betas.get('momentum', 0), 4),
                    size=round(optimized_betas.get('size', 0), 4)
                )
            )
        
        logger.info(f"Optimization completed successfully")
        
        return OptimizationResponse(
            optimization_strategy=request.strategy.value,
            allocation_changes=allocation_changes,
            factor_betas=factor_betas_response
        )
        
    except PortfolioOptimizerException as e:
        # Re-raise our own exceptions as-is since they already have the right
        # status code and structured detail — no need to wrap them again.
        logger.error(f"Portfolio optimizer exception: {e.detail}")
        raise
    except Exception as e:
        # Anything else is unexpected, so we wrap it in a generic 500 to avoid
        # leaking internal stack details to the client.
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise ErrorHandler.internal_error(f"Optimization failed: {str(e)}")

@router.get("/funds")
async def get_available_funds():
    # Loops through every loaded fund and attaches its name and dividend yield
    # so clients can build a picker UI without needing a separate info lookup per ticker.
    try:
        funds = []
        for ticker in data_loader.get_all_funds():
            info = data_loader.get_fund_info(ticker)
            funds.append({
                "ticker": ticker,
                "name": info.get('fund_name', ticker),
                "dividend_yield": info.get('dividend_yield', 'N/A')
            })
        
        return {"funds": funds, "count": len(funds)}
    except Exception as e:
        logger.error(f"Failed to get funds: {str(e)}")
        raise ErrorHandler.internal_error("Failed to retrieve fund information")

@router.get("/debug/data")
async def debug_data():
    # Development-only endpoint that dumps the full data summary so we can
    # quickly confirm what got loaded without having to restart the server and
    # dig through logs. Returns the error as a plain dict instead of raising
    # so it's always reachable even when data is in a bad state.
    try:
        summary = data_loader.get_data_summary()
        return summary
    except Exception as e:
        return {"error": str(e)}