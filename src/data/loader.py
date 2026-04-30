"""
Data loader module for fetching intraday market data from Yahoo Finance.

Supports:
- Multi-ticker downloads with retry logic
- Intraday intervals (1m, 5m, 15m, 60m, 1d)
- Local caching to reduce API calls
- Data validation and quality checks
"""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import yfinance as yf
import numpy as np


class DataLoader:
    """
    Loads and caches intraday market data from Yahoo Finance.
    
    Attributes:
        cache_dir: Directory for storing cached data
        interval: Intraday interval (default: '15m' for 15-minute bars)
        max_retries: Maximum retry attempts for failed downloads
    """

    def __init__(
        self,
        cache_dir: str = "./data/raw",
        interval: str = "15m",
        max_retries: int = 3,
    ):
        """
        Initialize DataLoader.
        
        Args:
            cache_dir: Directory to cache downloaded data
            interval: Intraday interval ('1m', '5m', '15m', '60m', '1d')
            max_retries: Number of retries for failed downloads
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.max_retries = max_retries
        
        # Validate interval
        valid_intervals = ["1m", "5m", "15m", "60m", "1d"]
        if interval not in valid_intervals:
            raise ValueError(f"Interval must be one of {valid_intervals}")

    def _get_cache_path(self, tickers: List[str], period: str) -> Path:
        """Generate cache file path from ticker list and period."""
        ticker_str = "_".join(sorted(tickers))
        filename = f"{ticker_str}_{self.interval}_{period}.pkl"
        return self.cache_dir / filename

    def _load_from_cache(self, tickers: List[str], period: str) -> Optional[pd.DataFrame]:
        """Load data from cache if it exists and is recent."""
        cache_path = self._get_cache_path(tickers, period)
        
        if not cache_path.exists():
            return None
        
        # Check cache age (refresh if older than 1 hour for intraday data)
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(hours=1) and self.interval != "1d":
            return None
        
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, data: pd.DataFrame, tickers: List[str], period: str) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(tickers, period)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def _validate_data(self, data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """
        Validate downloaded data quality.
        
        Args:
            data: Downloaded OHLCV data
            tickers: List of tickers
            
        Returns:
            Validated and cleaned data
        """
        if data is None or data.empty:
            raise ValueError("Downloaded data is empty")
        
        # Check for required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        
        # Handle MultiIndex columns (when multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers:
                for col in required_cols:
                    if (ticker, col) not in data.columns:
                        raise ValueError(f"Missing {col} for {ticker}")
        else:
            for col in required_cols:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
        
        # Forward fill missing data (up to 5 periods)
        data = data.fillna(method="ffill", limit=5)
        
        # Drop remaining NaN rows
        data = data.dropna()
        
        if data.empty:
            raise ValueError("Data became empty after validation")
        
        return data

    def fetch(
        self,
        tickers: List[str],
        period: str = "7d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
            period: Data period ('1d', '5d', '7d', '30d', '60d', '1y')
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with MultiIndex columns (ticker, OHLCV)
            
        Raises:
            ValueError: If data validation fails
        """
        # Validate inputs
        if not tickers:
            raise ValueError("tickers list cannot be empty")
        
        tickers = [t.upper() for t in tickers]
        
        # Try loading from cache
        if use_cache:
            cached_data = self._load_from_cache(tickers, period)
            if cached_data is not None:
                print(f"Loaded {len(tickers)} tickers from cache")
                return cached_data
        
        # Download data with retries
        print(f"Downloading {len(tickers)} tickers for period {period} with {self.interval} interval...")
        
        for attempt in range(self.max_retries):
            try:
                data = yf.download(
                    tickers,
                    period=period,
                    interval=self.interval,
                    progress=False,
                    threads=True,
                )
                
                # Validate data
                data = self._validate_data(data, tickers)
                
                # Save to cache
                if use_cache:
                    self._save_to_cache(data, tickers, period)
                
                print(f"Successfully downloaded {len(tickers)} tickers")
                return data
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Download attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    raise RuntimeError(f"Failed to download data after {self.max_retries} attempts: {e}")

    def fetch_single(
        self,
        ticker: str,
        period: str = "7d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Convenience method to fetch data for a single ticker.
        
        Args:
            ticker: Ticker symbol
            period: Data period
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume, Dividends, Stock Splits]
        """
        data = self.fetch([ticker.upper()], period, use_cache)
        
        # Flatten MultiIndex columns for single ticker
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(1)
        
        return data

    def clear_cache(self) -> None:
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("Cache cleared")

    def get_cache_info(self) -> Dict[str, object]:
        """Get information about cached files."""
        if not self.cache_dir.exists():
            return {}
        
        info = {}
        for cache_file in self.cache_dir.glob("*.pkl"):
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            info[cache_file.name] = {
                "modified": mod_time,
                "size_mb": round(size_mb, 2),
            }
        
        return info
