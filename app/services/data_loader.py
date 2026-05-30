import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from app.utils.errors import ErrorHandler

logger = logging.getLogger(__name__)

class DataLoader:
    
    def __init__(self, data_path: str = "data", filename: str = "data.xlsx"):
        # Build the full file path from the folder and filename so every method
        # in this class can just reference self.file_path without reconstructing it.
        # We also initialize all the data holders to None so it's obvious at a glance
        # what data this class is responsible for loading.
        self.data_path = Path(data_path)
        self.filename = filename
        self.file_path = self.data_path / filename
        
        self.fund_returns = None
        self.fund_info = None
        self.factor_returns = None
        self.dividend_yields = {}
        
        self._load_data()
    
    def _load_data(self):
        # Entry point that orchestrates loading all three sheets in order.
        # We read sheet names first so we can log them and reference by position,
        # which is more reliable than hardcoding sheet names that might change.
        # Fund info must load before returns since we need the ticker list.
        # Factor returns are optional — we log a warning and move on if missing.
        try:
            if not self.file_path.exists():
                raise FileNotFoundError(f"Excel file not found: {self.file_path}")
            
            excel_file = pd.ExcelFile(self.file_path)
            sheet_names = excel_file.sheet_names
            logger.info(f"Loading data from {self.filename}. Sheets found: {sheet_names}")
            
            # Based on your Excel structure:
            # Sheet 0 (first sheet): Fund Info (ticker, fund_name, dividend_yield)
            # Sheet 1 (second sheet): Fund Returns (date, ticker, total_return)
            # Sheet 2 (third sheet): Factor Returns (date, total_return, index_ticker)
            
            if len(sheet_names) >= 1:
                logger.info(f"Loading Fund Info from sheet: {sheet_names[0]}")
                self._load_fund_info(excel_file, sheet_names[0])
            else:
                raise ValueError("No sheets found in Excel file")
            
            if len(sheet_names) >= 2:
                logger.info(f"Loading Fund Returns from sheet: {sheet_names[1]}")
                self._load_fund_returns(excel_file, sheet_names[1])
            else:
                raise ValueError("No fund returns sheet found")
            
            if len(sheet_names) >= 3:
                logger.info(f"Loading Factor Returns from sheet: {sheet_names[2]}")
                self._load_factor_returns(excel_file, sheet_names[2])
            else:
                logger.warning("No factor returns sheet found. Factor exposure optimization will not work.")
                self.factor_returns = None
                
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise ErrorHandler.data_load_error(f"Failed to load Excel file: {str(e)}")
    
    def _load_fund_info(self, excel_file: pd.ExcelFile, sheet_name: str):
        # We don't assume fixed column names — instead we scan each column header
        # and match by keyword so the loader still works if someone renames a column
        # slightly (e.g. "fund_name" vs "name"). If no ticker column is found at all
        # we fall back to using the first column rather than crashing.
        # Dividend yields come in as percentages (e.g. "3.28%"), so we strip the %
        # and divide by 100 to store them as decimals for the math downstream.
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            logger.info(f"Loading fund info from '{sheet_name}' with {len(df)} rows")
            logger.info(f"Columns: {list(df.columns)}")
            
            ticker_col = None
            name_col = None
            div_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'ticker' in col_lower:
                    ticker_col = col
                elif 'fund_name' in col_lower or 'name' in col_lower:
                    name_col = col
                elif 'dividend' in col_lower or 'yield' in col_lower:
                    div_col = col
            
            if ticker_col is None:
                logger.error(f"Could not find ticker column. Columns: {list(df.columns)}")
                ticker_col = df.columns[0]
                logger.info(f"Using first column '{ticker_col}' as ticker")
            
            for _, row in df.iterrows():
                ticker = str(row[ticker_col]).strip()
                if not ticker or pd.isna(ticker) or ticker == 'nan' or ticker == '':
                    continue
                
                if name_col and not pd.isna(row[name_col]):
                    fund_name = str(row[name_col]).strip()
                else:
                    fund_name = ticker
                
                div_yield = 0.0
                if div_col and not pd.isna(row[div_col]) and row[div_col] != '':
                    div_val = str(row[div_col]).strip()
                    if div_val and div_val != 'nan':
                        div_val = div_val.replace('%', '').strip()
                        try:
                            div_yield = float(div_val) / 100
                        except:
                            div_yield = 0.0
                
                self.dividend_yields[ticker] = div_yield
                
                if self.fund_info is None:
                    self.fund_info = pd.DataFrame(columns=['fund_name', 'dividend_yield'])
                
                self.fund_info.loc[ticker] = [fund_name, f"{div_yield*100:.2f}%"]
            
            logger.info(f"Loaded {len(self.dividend_yields)} funds: {list(self.dividend_yields.keys())}")
            
        except Exception as e:
            logger.error(f"Error loading fund info: {str(e)}")
            raise
    
    def _load_fund_returns(self, excel_file: pd.ExcelFile, sheet_name: str):
        # Raw data comes in long format (one row per date-ticker pair), so after cleaning
        # we pivot it into wide format where each column is a ticker and each row is a date.
        # Returns also come in as percentages so we divide by 100 before storing.
        # We forward-fill then backward-fill to handle any missing trading days cleanly
        # without introducing NaNs that would break the optimizer math later.
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            logger.info(f"Loading fund returns from '{sheet_name}' with {len(df)} rows")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"First 5 rows:\n{df.head()}")
            
            date_col = None
            ticker_col = None
            return_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'date' in col_lower:
                    date_col = col
                elif 'ticker' in col_lower:
                    ticker_col = col
                elif 'return' in col_lower or 'total_return' in col_lower:
                    return_col = col
            
            if not all([date_col, ticker_col, return_col]):
                raise ValueError(f"Missing required columns. Found date:{date_col}, ticker:{ticker_col}, return:{return_col}")
            
            df_clean = pd.DataFrame()
            df_clean['date'] = pd.to_datetime(df[date_col])
            df_clean['ticker'] = df[ticker_col].astype(str).str.strip()
            
            returns_raw = df[return_col].astype(str).str.replace('%', '').str.strip()
            df_clean['return'] = pd.to_numeric(returns_raw, errors='coerce')
            df_clean = df_clean.dropna(subset=['return'])
            
            unique_tickers = df_clean['ticker'].unique()
            logger.info(f"Unique tickers found in returns data: {list(unique_tickers)}")
            
            pivot_df = df_clean.pivot(index='date', columns='ticker', values='return')
            pivot_df = pivot_df.dropna(how='all')
            
            self.fund_returns = pivot_df
            
            logger.info(f"Processed fund returns: {len(self.fund_returns)} days, {len(self.fund_returns.columns)} funds")
            logger.info(f"Funds available: {list(self.fund_returns.columns)}")
            
            if len(self.fund_returns) > 0:
                logger.info(f"Date range: {self.fund_returns.index[0]} to {self.fund_returns.index[-1]}")
            
        except Exception as e:
            logger.error(f"Error loading fund returns: {str(e)}")
            raise
    
    def _load_factor_returns(self, excel_file: pd.ExcelFile, sheet_name: str):
        # Same long-to-wide pivot pattern as fund returns but for factors.
        # Factor names get lowercased and spaces replaced with underscores so they
        # match the factor keys used throughout the rest of the codebase.
        # If anything goes wrong here we log a warning and set factor_returns to None
        # rather than raising — factor data is optional and we don't want it to block startup.
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            logger.info(f"Loading factor returns from '{sheet_name}' with {len(df)} rows")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"First 5 rows:\n{df.head()}")
            
            date_col = None
            factor_col = None
            return_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'date' in col_lower:
                    date_col = col
                elif 'index_ticker' in col_lower or 'factor' in col_lower:
                    factor_col = col
                elif 'return' in col_lower or 'total_return' in col_lower:
                    return_col = col
            
            if not all([date_col, factor_col, return_col]):
                logger.warning(f"Factor columns not found. Found date:{date_col}, factor:{factor_col}, return:{return_col}")
                return
            
            df_clean = pd.DataFrame()
            df_clean['date'] = pd.to_datetime(df[date_col])
            df_clean['factor'] = df[factor_col].astype(str).str.lower()
            df_clean['factor'] = df_clean['factor'].str.replace(' factor', '').str.replace(' ', '_').str.strip()
            
            returns_raw = df[return_col].astype(str).str.replace('%', '').str.strip()
            df_clean['return'] = pd.to_numeric(returns_raw, errors='coerce')
            df_clean = df_clean.dropna(subset=['return'])
            
            unique_factors = df_clean['factor'].unique()
            logger.info(f"Unique factors found: {list(unique_factors)}")
            
            pivot_df = pivot_df.dropna(how='all')
            pivot_df = pivot_df.ffill().bfill()
            
            self.factor_returns = pivot_df
            
            logger.info(f"Processed factor returns: {len(self.factor_returns)} days, factors: {list(self.factor_returns.columns)}")
            
            if len(self.factor_returns) > 0:
                logger.info(f"Date range: {self.factor_returns.index[0]} to {self.factor_returns.index[-1]}")
            
        except Exception as e:
            logger.error(f"Error loading factor returns: {str(e)}")
            self.factor_returns = None
            logger.warning("Factor returns not available. Factor exposure optimization will not work.")
    
    def _create_fund_info_from_returns(self):
        # Fallback used when the fund info sheet is missing or fails to load.
        if self.fund_returns is None:
            return
        
        tickers = self.fund_returns.columns
        self.fund_info = pd.DataFrame(index=tickers)
        self.fund_info['fund_name'] = tickers
        self.fund_info['dividend_yield'] = '0%'
        
        for ticker in tickers:
            if ticker not in self.dividend_yields:
                self.dividend_yields[ticker] = 0.0
        
        logger.info(f"Created fund info from returns data for {len(tickers)} funds")
    
    def get_fund_returns(self, tickers: List[str]) -> pd.DataFrame:
        # Validate that all requested tickers actually exist in our loaded data
        # before slicing — a clear error here saves a confusing KeyError later.
        if self.fund_returns is None:
            raise ErrorHandler.data_load_error("Fund returns data not loaded")
        
        logger.info(f"Available funds in data: {list(self.fund_returns.columns)}")
        logger.info(f"Requested tickers: {tickers}")
        
        available_tickers = set(self.fund_returns.columns)
        missing = set(tickers) - available_tickers
        
        if missing:
            raise ErrorHandler.invalid_ticker(
                f"Tickers not found: {missing}. Available: {available_tickers}"
            )
        
        result = self.fund_returns[tickers].copy()
        
        if len(result) == 0:
            raise ErrorHandler.infeasible_constraints(
                f"No return data available for tickers: {tickers}"
            )
        
        logger.info(f"Returning {len(result)} days of data for {tickers}")
        return result

    def get_fund_returns_for_date_range(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        # Fetches returns for the given tickers and then slices to the requested
        # date window. Both start_date and end_date are optional — if only one is
        # provided we apply just that boundary and leave the other end open.
        # After slicing we validate that enough rows remain (at least 30 trading days)
        # to produce a meaningful optimization result.
        # This is how the API mirrors the live tool's Time Frame setting —
        # the same underlying data is used but viewed through a different window.
        returns = self.get_fund_returns(tickers)
        # Drop rows where ANY ticker has NaN or zero (e.g. before a fund existed)
        # This ensures we only use the true common date range across all tickers
        returns = returns.replace(0, np.nan)
        returns = returns.dropna(how='any')
        logger.info(f"After cleaning zeros/NaN, date range: {returns.index[0]} to {returns.index[-1]}, {len(returns)} rows")
        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            returns = returns[returns.index >= start_dt]
            logger.info(f"Applied start_date filter ({start_date}): {len(returns)} rows remain")

        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            returns = returns[returns.index <= end_dt]
            logger.info(f"Applied end_date filter ({end_date}): {len(returns)} rows remain")

        if len(returns) < 30:
            raise ErrorHandler.infeasible_constraints(
                f"After applying date range ({start_date} → {end_date}), only {len(returns)} "
                f"trading days remain. Need at least 30 days for a valid optimization."
            )

        return returns
    
    def get_fund_info(self, ticker: str) -> Dict:
        # Simple lookup — returns a safe default dict if the ticker isn't in fund_info
        # so callers never have to handle a KeyError or None check.
        if self.fund_info is not None and ticker in self.fund_info.index:
            return {
                'fund_name': self.fund_info.loc[ticker, 'fund_name'],
                'dividend_yield': self.fund_info.loc[ticker, 'dividend_yield']
            }
        return {'fund_name': ticker, 'dividend_yield': 'N/A'}
    
    def get_all_funds(self) -> List[str]:
        # Returns column names from fund_returns which is the authoritative list of
        # tickers we actually have price history for.
        if self.fund_returns is not None:
            return list(self.fund_returns.columns)
        return []
    
    def get_factor_returns(self) -> Optional[pd.DataFrame]:
        # Thin accessor so callers don't touch self.factor_returns directly.
        return self.factor_returns
    
    def get_dividend_yield(self, ticker: str) -> float:
        # Returns 0.0 as the default so math downstream never breaks on a missing ticker.
        return self.dividend_yields.get(ticker, 0.0)
    
    def get_common_date_range(self, returns: pd.DataFrame) -> pd.DataFrame:
        # Factor returns and fund returns may not cover identical date ranges,
        # so we intersect their indexes and return only the overlapping dates.
        if self.factor_returns is None:
            return returns
        
        common_dates = returns.index.intersection(self.factor_returns.index)
        if len(common_dates) > 0:
            logger.info(f"Found {len(common_dates)} common dates between returns and factors")
            return returns.loc[common_dates]
        
        logger.warning("No common dates found between fund returns and factor returns")
        return returns
    
    def get_data_summary(self) -> Dict[str, Any]:
        # Aggregates everything we know about the loaded data into one dict.
        summary = {
            'fund_returns': {
                'loaded': self.fund_returns is not None,
                'days': len(self.fund_returns) if self.fund_returns is not None else 0,
                'funds': list(self.fund_returns.columns) if self.fund_returns is not None else [],
                'date_range': {
                    'start': str(self.fund_returns.index[0]) if self.fund_returns is not None and len(self.fund_returns) > 0 else None,
                    'end': str(self.fund_returns.index[-1]) if self.fund_returns is not None and len(self.fund_returns) > 0 else None
                }
            },
            'fund_info': {
                'loaded': self.fund_info is not None,
                'funds': list(self.fund_info.index) if self.fund_info is not None else []
            },
            'factor_returns': {
                'loaded': self.factor_returns is not None,
                'days': len(self.factor_returns) if self.factor_returns is not None else 0,
                'factors': list(self.factor_returns.columns) if self.factor_returns is not None else [],
                'date_range': {
                    'start': str(self.factor_returns.index[0]) if self.factor_returns is not None and len(self.factor_returns) > 0 else None,
                    'end': str(self.factor_returns.index[-1]) if self.factor_returns is not None and len(self.factor_returns) > 0 else None
                }
            },
            'dividend_yields': self.dividend_yields,
            'file_path': str(self.file_path),
            'file_exists': self.file_path.exists(),
            'sheet_count': len(pd.ExcelFile(self.file_path).sheet_names) if self.file_path.exists() else 0
        }
        return summary