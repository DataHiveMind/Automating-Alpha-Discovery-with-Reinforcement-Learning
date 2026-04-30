# Data Module Documentation

## Overview

The `src/data` module provides a complete, production-ready pipeline for fetching intraday market data from Yahoo Finance, validating data quality, and preprocessing it for the RL agent's state space.

## Architecture

```
src/data/
├── __init__.py              # Module exports
├── loader.py                # DataLoader - Yahoo Finance integration
├── preprocessor.py          # DataPreprocessor - Feature engineering & transformations
├── validator.py             # DataValidator - Data quality checks
└── example.py               # Example usage and integration patterns
```

## Components

### 1. DataLoader (`loader.py`)

Fetches intraday OHLCV data from Yahoo Finance with caching and retry logic.

#### Key Features:
- **Multi-ticker support**: Download data for multiple assets simultaneously
- **Intraday intervals**: 1m, 5m, 15m, 60m, 1d
- **Intelligent caching**: Automatically caches data to reduce API calls
- **Retry logic**: Handles transient download failures
- **Data validation**: Checks for missing data and OHLCV consistency

#### Usage:

```python
from src.data import DataLoader

# Initialize loader
loader = DataLoader(interval="15m", cache_dir="./data/raw")

# Fetch multi-ticker data
data = loader.fetch(
    tickers=["AAPL", "MSFT", "GOOGL"],
    period="7d",
    use_cache=True
)

# Or fetch single ticker
single = loader.fetch_single("AAPL", period="30d")

# Manage cache
loader.clear_cache()
cache_info = loader.get_cache_info()
```

#### DataFrame Format:

For multiple tickers, returns MultiIndex columns:
```
                           AAPL                         MSFT
                Open  High   Low  Close Volume  Open  High  ...
Datetime
2024-01-15 10:00:00  150.5  151  149   150.8  1000000  ...
```

### 2. DataPreprocessor (`preprocessor.py`)

Transforms raw market data into features required by the RL environment.

#### Time-Series Operations:

```python
from src.data import DataPreprocessor

preprocessor = DataPreprocessor(lookback_window=20)

# Rolling mean
ts_mean = preprocessor.ts_mean(price_series, window=20)

# Rolling std
ts_std = preprocessor.ts_std(price_series, window=20)

# Delay (lag)
lagged = preprocessor.ts_delay(price_series, periods=1)

# Rolling correlation
corr = preprocessor.ts_corr(price_series, volume_series, window=20)

# Period-to-period change (returns)
returns = preprocessor.ts_delta(price_series, periods=1)
```

#### Cross-Sectional Operations:

```python
# Cross-sectional ranking (0-1 percentile)
ranked = preprocessor.cs_rank(data_df, axis=1, pct=True)

# Cross-sectional normalization
normalized = preprocessor.cs_normalize(data_df, method="zscore")  # or "minmax", "mad"
```

#### RL State Preparation:

```python
# Prepare normalized state vector for RL agent
normalized_returns, market_entropy = preprocessor.prepare_rl_state(
    data=ohlcv_data,
    price_col="Close",
    volume_col="Volume"
)
# Returns: (state_array, entropy_array)
```

#### Full Pipeline:

```python
# Process batch with all engineered features
processed = preprocessor.process_batch(
    data=ohlcv_data,
    price_col="Close",
    normalize=True
)
# Creates: log_returns, ts_mean, ts_std, price_delay_1, price_delay_5, volume_mean, volume_std
```

#### Market Entropy:

```python
# Calculate cross-sectional market entropy (volatility dispersion)
entropy = preprocessor.calculate_market_entropy(
    returns=returns_df,
    window=20
)
# Useful for RL agent awareness of market stress/stability
```

#### Resampling:

```python
# Resample intraday data to daily
daily_data = preprocessor.resample_data(
    data=intraday_data,
    freq="1D",
    agg_method="ohlc"  # or "mean", "first", "last"
)
```

### 3. DataValidator (`validator.py`)

Comprehensive data quality assurance and anomaly detection.

#### Validation Checks:

```python
from src.data.validator import DataValidator

validator = DataValidator()

# Run full validation
report = validator.full_validation(data, price_col="Close")

# Pretty print report
validator.print_validation_report(report)

# Individual checks
missing = validator.check_missing_data(data, threshold=0.1)
gaps = validator.check_price_gaps(data, gap_threshold=0.20)
vol_anomalies = validator.check_volume_anomalies(data, std_threshold=3.0)
ohlc = validator.check_ohlc_consistency(data)
returns_dist = validator.check_returns_distribution(data)
```

#### Report Structure:

```python
{
    "timestamp": "2024-01-15T10:30:00",
    "total_rows": 1000,
    "date_range": {"start": "...", "end": "..."},
    "missing_data": {"passed": True, ...},
    "price_gaps": {"passed": False, "total_gaps_detected": 2, ...},
    "volume_anomalies": {"passed": True, ...},
    "ohlc_consistency": {"passed": True, ...},
    "returns_distribution": {"passed": True, ...},
    "overall_passed": True
}
```

## Integration with RL Pipeline

### Typical workflow:

```python
from src.data import DataLoader, DataPreprocessor
from src.data.validator import DataValidator

# 1. LOAD DATA
loader = DataLoader(interval="15m")
raw_data = loader.fetch(
    tickers=["AAPL", "MSFT", "GOOGL"],
    period="30d"
)

# 2. VALIDATE
validator = DataValidator()
report = validator.full_validation(raw_data.iloc[:, 0:5])  # Check first ticker
if not report["overall_passed"]:
    print("Data validation failed!")
    exit(1)

# 3. PREPROCESS
preprocessor = DataPreprocessor(lookback_window=20)
processed = preprocessor.process_batch(raw_data)

# 4. PREPARE RL STATE
state, entropy = preprocessor.prepare_rl_state(raw_data)

# 5. USE IN ENVIRONMENT
# state and entropy are ready for your Gymnasium environment
```

## Configuration

### Intervals:
- `"1m"`: 1-minute bars (7 days max per request)
- `"5m"`: 5-minute bars (60 days max)
- `"15m"`: 15-minute bars (60 days max)
- `"60m"`: Hourly bars (730 days max)
- `"1d"`: Daily bars (unlimited)

### Lookback Window:
Default is 20 periods. Adjust based on your time-series window needs:
```python
preprocessor = DataPreprocessor(lookback_window=50)  # Longer window
```

### Normalization Methods:
- `"zscore"`: (x - mean) / std
- `"minmax"`: (x - min) / (max - min)
- `"mad"`: (x - median) / median_absolute_deviation

## Performance Considerations

1. **Caching**: First call downloads data, subsequent calls use cache (1-hour TTL for intraday)
2. **Memory**: For 1000+ tickers × 100+ days of 1-min bars, consider batching
3. **Yahoo Finance Rate Limits**: Generally ~2000 requests/hour; batching helps
4. **Local Storage**: Each 7-day, 4-ticker dataset ≈ 10-20MB cached

## Error Handling

```python
try:
    data = loader.fetch(["AAPL"], period="7d")
except ValueError as e:
    print(f"Invalid parameters: {e}")
except RuntimeError as e:
    print(f"Download failed after retries: {e}")
```

## Examples

See `src/data/example.py` for comprehensive examples:
- Basic data loading
- Data preprocessing
- Time-series operations
- Cross-sectional operations
- Data validation
- Resampling
- Full integration pipeline

Run examples:
```bash
cd src/data
python example.py
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Key packages:
- `yfinance`: Yahoo Finance data fetching
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scipy`: Statistical functions

## Future Enhancements

- [ ] Support for other data sources (Alpaca, IB, etc.)
- [ ] Real-time streaming data
- [ ] Fundamental data integration
- [ ] Sentiment/sentiment scores
- [ ] Alternative data sources (crypto, commodities)
- [ ] Async/concurrent downloading
