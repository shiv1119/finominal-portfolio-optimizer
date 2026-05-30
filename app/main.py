from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.utils.errors import PortfolioOptimizerException, ErrorCode
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    # Title and description show up in the auto-generated /docs Swagger UI,
    # so future developers (or clients) can understand what this API does
    # without reading the source code.
    title="Portfolio Optimizer API",
    description="Advanced portfolio optimization API with multiple strategies",
    version="1.0.0"
)

app.add_middleware(
    # CORS middleware allows the API to be called from a browser running on a
    # different origin (e.g. a React frontend on localhost:3000 hitting this on :8000).
    # allow_origins=["*"] is wide open for development — tighten this to specific
    # domains before deploying to production.
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["portfolio"])
# All routes defined in routes.py are mounted under /api/v1 so we can introduce
# a /api/v2 later without breaking existing clients.

@app.exception_handler(PortfolioOptimizerException)
async def portfolio_optimizer_exception_handler(request: Request, exc: PortfolioOptimizerException):
    # Intercepts any PortfolioOptimizerException raised anywhere in the app and
    # serializes it into a JSON response using the status code and structured detail
    # we already set when the exception was created — no extra formatting needed here.
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Catches anything that slipped past the specific handler above.
    # We always log the full error server-side, but only expose the raw error
    # string to the client when debug mode is on — in production details stays
    # empty so we don't leak internal information.
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error_code": ErrorCode.INTERNAL_ERROR,
            "message": "An unexpected error occurred",
            "details": {"error": str(exc)} if app.debug else {}
        }
    )

@app.get("/")
async def root():
    # Simple landing response that points newcomers to the right places —
    # /docs for the interactive Swagger UI and /api/v1/health to confirm the
    # service is running with data loaded.
    return {
        "message": "Portfolio Optimizer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }
