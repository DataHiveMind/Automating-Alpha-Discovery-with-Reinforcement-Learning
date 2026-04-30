"""
Data preprocessing module for cross-sectional and time-series transformations.

Provides operations required by the RL environment:
- Time-Series: ts_mean, ts_std, delay, ts_corr
- Cross-Sectional: cs_rank, cs_normalize
- Market entropy calculations
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


class DataPreprocessor:
    """
    Preprocesses market data for RL agent state space.
    
    Handles cross-sectional normalization, time-series feature engineering,
    and market entropy calculations.
    """

    def __init__(self, lookback_window: int = 20):
        """
        Initialize DataPreprocessor.
        
        Args:
            lookback_window: Window size for time-series calculations (default: 20 periods)
        """
        self.lookback_window = lookback_window

    # ========== TIME-SERIES OPERATIONS ==========

    def ts_mean(
        self,
        data: pd.Series,
        window: Optional[int] = None,
        min_periods: int = 1,
    ) -> pd.Series:
        """
        Calculate rolling time-series mean.
        
        Args:
            data: Time series data
            window: Window size (default: lookback_window)
            min_periods: Minimum periods for calculation
            
        Returns:
            Rolling mean series
        """
        window = window or self.lookback_window
        return data.rolling(window=window, min_periods=min_periods).mean()

    def ts_std(
        self,
        data: pd.Series,
        window: Optional[int] = None,
        min_periods: int = 1,
    ) -> pd.Series:
        """
        Calculate rolling time-series standard deviation.
        
        Args:
            data: Time series data
            window: Window size (default: lookback_window)
            min_periods: Minimum periods for calculation
            
        Returns:
            Rolling standard deviation series
        """
        window = window or self.lookback_window
        return data.rolling(window=window, min_periods=min_periods).std()

    def ts_delay(
        self,
        data: pd.Series,
        periods: int = 1,
    ) -> pd.Series:
        """
        Delay (lag) a time series.
        
        Args:
            data: Time series data
            periods: Number of periods to lag (positive = past values)
            
        Returns:
            Lagged series
        """
        return data.shift(periods=periods)

    def ts_corr(
        self,
        data1: pd.Series,
        data2: pd.Series,
        window: Optional[int] = None,
        min_periods: int = 2,
    ) -> pd.Series:
        """
        Calculate rolling correlation between two time series.
        
        Args:
            data1: First time series
            data2: Second time series
            window: Window size (default: lookback_window)
            min_periods: Minimum periods for calculation
            
        Returns:
            Rolling correlation series
        """
        window = window or self.lookback_window
        return data1.rolling(window=window, min_periods=min_periods).corr(data2)

    def ts_delta(
        self,
        data: pd.Series,
        periods: int = 1,
    ) -> pd.Series:
        """
        Calculate period-to-period change (returns).
        
        Args:
            data: Time series data
            periods: Number of periods for difference
            
        Returns:
            Differenced series
        """
        return data.diff(periods=periods)

    # ========== CROSS-SECTIONAL OPERATIONS ==========

    def cs_rank(
        self,
        data: pd.DataFrame,
        axis: int = 1,
        pct: bool = True,
    ) -> pd.DataFrame:
        """
        Cross-sectional rank normalization.
        
        Args:
            data: DataFrame with assets as columns, time as index
            axis: 0 for time-series ranking, 1 for cross-sectional
            pct: Return percentile rank (0-1) instead of ordinal rank
            
        Returns:
            Ranked data
        """
        if pct:
            return data.rank(axis=axis, pct=True)
        return data.rank(axis=axis)

    def cs_normalize(
        self,
        data: pd.DataFrame,
        axis: int = 1,
        method: str = "zscore",
    ) -> pd.DataFrame:
        """
        Cross-sectional normalization.
        
        Args:
            data: DataFrame with assets as columns, time as index
            axis: 0 for time-series, 1 for cross-sectional
            method: 'zscore' (z-normalization), 'minmax' (min-max scaling), 'mad' (median absolute deviation)
            
        Returns:
            Normalized data
        """
        if method == "zscore":
            return (data - data.mean(axis=axis, keepdims=True)) / (data.std(axis=axis, keepdims=True) + 1e-8)
        
        elif method == "minmax":
            min_val = data.min(axis=axis, keepdims=True)
            max_val = data.max(axis=axis, keepdims=True)
            return (data - min_val) / (max_val - min_val + 1e-8)
        
        elif method == "mad":
            # Median Absolute Deviation
            median = data.median(axis=axis, keepdims=True)
            mad = (data - median).abs().median(axis=axis, keepdims=True)
            return (data - median) / (mad + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    # ========== MARKET ENTROPY & STATE PREPARATION ==========

    def calculate_market_entropy(
        self,
        returns: pd.DataFrame,
        window: Optional[int] = None,
    ) -> pd.Series:
        """
        Calculate market entropy (volatility dispersion across assets).
        
        Entropy is measured as the cross-sectional dispersion of returns,
        indicating market stress/stability for RL agent awareness.
        
        Args:
            returns: DataFrame of asset returns
            window: Window for rolling entropy
            
        Returns:
            Market entropy series
        """
        window = window or self.lookback_window
        
        # Cross-sectional volatility (standard deviation of returns across assets)
        def entropy_func(x):
            # Approximate entropy using coefficient of variation
            return x.std() / (np.abs(x.mean()) + 1e-8)
        
        entropy = returns.rolling(window=window, min_periods=1).apply(entropy_func, raw=False, axis=1)
        return entropy.iloc[:, 0] if isinstance(entropy, pd.DataFrame) else entropy

    def prepare_rl_state(
        self,
        data: pd.DataFrame,
        price_col: str = "Close",
        volume_col: str = "Volume",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare state vector for RL agent.
        
        Combines price returns and market entropy into normalized state space.
        
        Args:
            data: OHLCV data
            price_col: Column name for price data
            volume_col: Column name for volume data
            
        Returns:
            Tuple of (normalized_returns, market_entropy)
        """
        # Calculate returns
        if price_col in data.columns:
            prices = data[price_col]
            returns = prices.pct_change().fillna(0)
        else:
            raise ValueError(f"Column '{price_col}' not found in data")
        
        # Normalize returns
        normalized_returns = self.cs_normalize(
            returns.to_frame(),
            axis=0,
            method="zscore"
        ).values.flatten()
        
        # Calculate market entropy
        if isinstance(data, pd.DataFrame) and len(data.columns) > 1:
            returns_df = data.pct_change().fillna(0)
            entropy = self.calculate_market_entropy(returns_df)
        else:
            entropy = np.zeros(len(returns))
        
        return normalized_returns, entropy.values if hasattr(entropy, 'values') else entropy

    def process_batch(
        self,
        data: pd.DataFrame,
        price_col: str = "Close",
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline for a batch of data.
        
        Args:
            data: Raw OHLCV data
            price_col: Price column for preprocessing
            normalize: Whether to normalize data
            
        Returns:
            Preprocessed DataFrame with engineered features
        """
        processed = data.copy()
        
        # Calculate log returns (more numerically stable)
        if price_col in processed.columns:
            processed['log_returns'] = np.log(processed[price_col] / processed[price_col].shift(1))
            processed['log_returns'] = processed['log_returns'].fillna(0)
        
        # Time-series features
        if price_col in processed.columns:
            processed['ts_mean'] = self.ts_mean(processed[price_col])
            processed['ts_std'] = self.ts_std(processed[price_col])
            processed['price_delay_1'] = self.ts_delay(processed[price_col], periods=1)
            processed['price_delay_5'] = self.ts_delay(processed[price_col], periods=5)
        
        # Volume features
        if 'Volume' in processed.columns:
            processed['volume_mean'] = self.ts_mean(processed['Volume'])
            processed['volume_std'] = self.ts_std(processed['Volume'])
        
        # Drop NaN rows created by rolling operations
        processed = processed.dropna()
        
        # Normalize if requested
        if normalize:
            for col in ['log_returns', 'ts_mean', 'ts_std', 'volume_mean', 'volume_std']:
                if col in processed.columns:
                    processed[col] = self.cs_normalize(
                        processed[[col]],
                        axis=0,
                        method="zscore"
                    )
        
        return processed

    # ========== UTILITY METHODS ==========

    def resample_data(
        self,
        data: pd.DataFrame,
        freq: str = "1D",
        agg_method: str = "ohlc",
    ) -> pd.DataFrame:
        """
        Resample intraday data to different frequency.
        
        Args:
            data: OHLCV data with datetime index
            freq: Target frequency ('1H', '1D', '1W', etc.)
            agg_method: Aggregation method ('ohlc', 'mean', 'first', 'last')
            
        Returns:
            Resampled data
        """
        if agg_method == "ohlc":
            agg_dict = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
            }
            return data.resample(freq).agg(agg_dict).dropna()
        
        elif agg_method == "mean":
            return data.resample(freq).mean().dropna()
        
        else:
            return data.resample(freq).agg(agg_method).dropna()

    def handle_missing_data(
        self,
        data: pd.DataFrame,
        method: str = "forward_fill",
        limit: int = 5,
    ) -> pd.DataFrame:
        """
        Handle missing data points.
        
        Args:
            data: Data with potential gaps
            method: 'forward_fill', 'interpolate', 'drop'
            limit: Maximum consecutive missing values to fill
            
        Returns:
            Data with missing values handled
        """
        if method == "forward_fill":
            return data.fillna(method="ffill", limit=limit).fillna(method="bfill")
        
        elif method == "interpolate":
            return data.interpolate(method="linear", limit=limit)
        
        elif method == "drop":
            return data.dropna()
        
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_asset_returns(
        self,
        data: pd.DataFrame,
        price_col: str = "Close",
        log_returns: bool = True,
    ) -> pd.DataFrame:
        """
        Extract and calculate asset returns.
        
        Args:
            data: OHLCV data
            price_col: Price column to use
            log_returns: Use log returns instead of simple returns
            
        Returns:
            Returns DataFrame
        """
        if price_col not in data.columns:
            raise ValueError(f"Column '{price_col}' not found")
        
        if log_returns:
            returns = np.log(data[price_col] / data[price_col].shift(1))
        else:
            returns = data[price_col].pct_change()
        
        return returns.fillna(0)
