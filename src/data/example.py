"""
Example and integration script for the data module.

Demonstrates how to use DataLoader and DataPreprocessor for the RL pipeline.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from loader import DataLoader
from preprocessor import DataPreprocessor
from validator import DataValidator


def example_basic_usage():
    """Basic example: Load and preprocess data."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Data Loading and Preprocessing")
    print("="*60)
    
    # Initialize loader for intraday 15-min bars
    loader = DataLoader(interval="15m", cache_dir="./data/raw")
    
    # Download 7 days of intraday data for 3 tech stocks
    tickers = ["AAPL", "MSFT", "GOOGL"]
    data = loader.fetch(tickers, period="7d", use_cache=True)
    
    print(f"\nData shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{data.head()}")
    
    return data


def example_preprocessing(data):
    """Example: Preprocess data for RL agent."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Data Preprocessing")
    print("="*60)
    
    preprocessor = DataPreprocessor(lookback_window=20)
    
    # For single ticker (flatten multiindex)
    if isinstance(data.columns, pd.MultiIndex):
        ticker = data.columns.get_level_values(0)[0]
        ticker_data = data[ticker].copy()
    else:
        ticker_data = data.copy()
    
    # Process batch with all features
    processed = preprocessor.process_batch(ticker_data, normalize=True)
    
    print(f"\nProcessed data shape: {processed.shape}")
    print(f"Features created: {[c for c in processed.columns if c not in ticker_data.columns]}")
    print(f"\nProcessed data (first 5 rows):\n{processed.head()}")
    
    # Calculate RL state vector
    state, entropy = preprocessor.prepare_rl_state(ticker_data)
    print(f"\nRL State Vector shape: {state.shape}")
    print(f"Market Entropy shape: {entropy.shape}")
    print(f"State sample: {state[:5]}")
    print(f"Entropy sample: {entropy[:5]}")
    
    return processed


def example_time_series_operations():
    """Example: Time-series operations."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Time-Series Operations")
    print("="*60)
    
    loader = DataLoader(interval="15m")
    data = loader.fetch(["AAPL"], period="5d")
    
    if isinstance(data.columns, pd.MultiIndex):
        price = data["AAPL"]["Close"]
    else:
        price = data["Close"]
    
    preprocessor = DataPreprocessor(lookback_window=10)
    
    # Demonstrate time-series operations
    mean = preprocessor.ts_mean(price, window=10)
    std = preprocessor.ts_std(price, window=10)
    delayed = preprocessor.ts_delay(price, periods=1)
    delta = preprocessor.ts_delta(price, periods=1)
    
    print(f"\nOriginal Price (last 5): \n{price.tail()}")
    print(f"\nRolling Mean (last 5): \n{mean.tail()}")
    print(f"\nRolling Std (last 5): \n{std.tail()}")
    print(f"\nDelayed Price (last 5): \n{delayed.tail()}")
    print(f"\nPrice Delta/Returns (last 5): \n{delta.tail()}")


def example_cross_sectional_operations():
    """Example: Cross-sectional operations."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Cross-Sectional Operations")
    print("="*60)
    
    loader = DataLoader(interval="15m")
    data = loader.fetch(["AAPL", "MSFT", "GOOGL"], period="5d")
    
    # Extract closes (multi-ticker)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data.xs("Close", level=1, axis=1)
    else:
        closes = data[["Close"]] if "Close" in data.columns else data
    
    preprocessor = DataPreprocessor()
    
    # Cross-sectional rank
    ranked = preprocessor.cs_rank(closes, pct=True)
    print(f"\nCross-Sectional Rank (last 5):\n{ranked.tail()}")
    
    # Cross-sectional normalization
    normalized = preprocessor.cs_normalize(closes, method="zscore")
    print(f"\nCross-Sectional Normalized (last 5):\n{normalized.tail()}")
    
    # Calculate returns
    returns = closes.pct_change()
    
    # Market entropy
    entropy = preprocessor.calculate_market_entropy(returns)
    print(f"\nMarket Entropy (last 5):\n{entropy.tail()}")


def example_validation():
    """Example: Data validation."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Data Validation")
    print("="*60)
    
    loader = DataLoader(interval="1h")
    data = loader.fetch(["AAPL"], period="30d")
    
    if isinstance(data.columns, pd.MultiIndex):
        data = data["AAPL"]
    
    validator = DataValidator()
    report = validator.full_validation(data)
    
    validator.print_validation_report(report)


def example_resampling():
    """Example: Resample intraday data to daily."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Data Resampling")
    print("="*60)
    
    loader = DataLoader(interval="15m")
    intraday = loader.fetch(["AAPL"], period="10d")
    
    if isinstance(intraday.columns, pd.MultiIndex):
        intraday = intraday["AAPL"]
    
    preprocessor = DataPreprocessor()
    
    print(f"Intraday data shape: {intraday.shape}")
    print(f"Intraday freq: 15-min bars")
    
    # Resample to daily
    daily = preprocessor.resample_data(intraday, freq="1D", agg_method="ohlc")
    print(f"\nDaily data shape: {daily.shape}")
    print(f"Daily data (last 5):\n{daily.tail()}")


def example_integration():
    """Full integration example: Load, validate, preprocess."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Full Integration Pipeline")
    print("="*60)
    
    # Load data
    loader = DataLoader(interval="15m", cache_dir="./data/raw")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    print(f"Loading {len(tickers)} tickers...")
    data = loader.fetch(tickers, period="7d", use_cache=True)
    
    # Validate
    print("\nValidating data...")
    if isinstance(data.columns, pd.MultiIndex):
        aapl_data = data["AAPL"]
    else:
        aapl_data = data
    
    validator = DataValidator()
    report = validator.full_validation(aapl_data)
    validator.print_validation_report(report)
    
    # Preprocess
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor(lookback_window=20)
    processed = preprocessor.process_batch(aapl_data, normalize=True)
    
    print(f"Processed shape: {processed.shape}")
    print(f"Ready for RL agent!")
    
    # Get cache info
    print(f"\nCache info:")
    cache_info = loader.get_cache_info()
    for filename, info in cache_info.items():
        print(f"  {filename}: {info}")


if __name__ == "__main__":
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█  Data Module Examples & Integration                   █")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    try:
        # Run examples
        data = example_basic_usage()
        example_preprocessing(data)
        example_time_series_operations()
        example_cross_sectional_operations()
        example_validation()
        example_resampling()
        example_integration()
        
        print("\n✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
