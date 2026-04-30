# Evaluation Module Documentation

## Overview

The `src/evaluation` module provides a complete backtesting and factor analysis framework for evaluating RL-generated alpha factors.

## Architecture

```
src/evaluation/
├── __init__.py              # Module exports
├── backtester.py            # CrossSectionalBacktester - quintile/decile analysis
├── factor_analysis.py       # FactorAnalyzer - Fama-French regression & orthogonalization
├── validator.py             # EvaluationValidator - robustness & significance tests
└── example.py               # Example usage and integration patterns
```

## Components

### 1. CrossSectionalBacktester (`backtester.py`)

Performs cross-sectional backtests on alpha factors.

#### Key Features:
- **Quintile/Decile Sorting**: Long-short portfolio construction
- **Information Coefficient (IC)**: Rank and Pearson correlation with forward returns
- **Turnover Analysis**: Calculate portfolio rebalancing costs
- **Risk Metrics**: Sharpe ratio, maximum drawdown, Calmar ratio
- **Return Attribution**: Per-quintile analysis and long-short spread

#### Usage:

```python
from src.evaluation import CrossSectionalBacktester

backtester = CrossSectionalBacktester(periods_per_year=252)

# Quintile analysis
quintile_results = backtester.quintile_sort(
    factors=factor_values_df,      # Shape: (time, assets)
    returns=forward_returns_df,    # Shape: (time, assets)
    lag=1                          # 1-period forward-looking
)

# Results structure
for q in range(1, 6):
    q_data = quintile_results[f"Q{q}"]
    print(f"Q{q} Mean Return: {q_data['mean_return']}")
    print(f"Q{q} Sharpe Ratio: {q_data['sharpe']}")

ls = quintile_results["LS_Spread"]  # Long-short spread
print(f"L/S Sharpe: {ls['sharpe']}")
print(f"L/S t-stat: {ls['t_stat']}")
```

#### Information Coefficient:

```python
# Rank IC (Spearman correlation)
ic_series = backtester.information_coefficient(
    factors=factors_df,
    returns=returns_df,
    lag=1,
    method="rank"  # or "pearson"
)

print(f"Mean IC: {ic_series.mean():.4f}")
print(f"Std IC: {ic_series.std():.4f}")
print(f"Hit Rate: {(ic_series > 0).mean():.2%}")
```

#### Turnover & Transaction Costs:

```python
# Calculate turnover over time
turnover_series = backtester.cumulative_turnover(portfolio_weights)

# Estimate transaction costs (10 bps bid-ask spread)
costs = backtester.transaction_costs(turnover_series, bid_ask_spread=0.001)

print(f"Mean Turnover: {turnover_series.mean():.2%}")
print(f"Avg Transaction Cost: {costs.mean():.4f}")
```

#### Performance Summary:

```python
# Comprehensive summary table
summary = backtester.factor_performance_table(
    factors=factors_df,
    returns=returns_df,
    lag=1
)

print(summary)
# Output:
#    IC_Mean  IC_Std  IC_t-stat  Q1_Return  Q5_Return  LS_Return  LS_Sharpe
# 0  0.0543  0.1203    4.5123      0.0001    0.0015     0.0014      1.2300
```

---

### 2. FactorAnalyzer (`factor_analysis.py`)

Analyzes factor exposures and orthogonalizes against known risk factors.

#### Key Features:
- **Fama-French Regression**: 3-factor and 5-factor models
- **Factor Orthogonalization**: Remove known factor exposures
- **Risk Attribution**: Decompose variance into factor vs idiosyncratic
- **Correlation Analysis**: Measure factor overlap
- **Comprehensive Reporting**: Full factor analysis summaries

#### Usage:

```python
from src.evaluation import FactorAnalyzer

analyzer = FactorAnalyzer()

# Load Fama-French factors (auto-downloads)
ff_factors = analyzer.fetch_fama_french_factors(freq="daily", model="3f")
```

#### Fama-French Regression:

```python
# Run 3-factor regression
results = analyzer.fama_french_regression(
    returns=alpha_factor_series,
    factors=ff_factors,
    include_constant=True
)

print(f"Alpha (annualized): {results['alpha_annualized']:.4%}")
print(f"Alpha t-stat: {results['t_stats']['Alpha']:.2f}")
print(f"Market Beta: {results['coefficients']['Mkt-RF']:.4f}")
print(f"SMB Beta: {results['coefficients']['SMB']:.4f}")
print(f"HML Beta: {results['coefficients']['HML']:.4f}")
print(f"R-squared: {results['r_squared']:.4f}")
```

#### Factor Orthogonalization:

```python
# Remove market, size, and value exposures
orthogonal_factor = analyzer.orthogonalize_factor(
    factor=alpha_factor,
    control_factors=ff_factors[["Mkt-RF", "SMB", "HML"]],
    method="regression"  # or "gram_schmidt"
)

# Compare
corr_orig = alpha_factor.corr(ff_factors["Mkt-RF"])
corr_ortho = orthogonal_factor.corr(ff_factors["Mkt-RF"])
print(f"Correlation with Mkt-RF: {corr_orig:.4f} → {corr_ortho:.4f}")
```

#### Variance Decomposition:

```python
# Break down where factor variance comes from
variance_breakdown = analyzer.idiosyncratic_variance(
    alpha_factor=alpha_factor,
    known_factors=ff_factors
)

print(f"Explained by Fama-French: {variance_breakdown['explained_variance']:.6f}")
print(f"Idiosyncratic (novel): {variance_breakdown['idiosyncratic_variance']:.6f}")
print(f"Idiosyncratic %: {variance_breakdown['idiosyncratic_pct']:.1f}%")
```

#### Correlation Matrix:

```python
# See factor overlaps
corr_matrix = analyzer.factor_correlation_matrix(
    alpha_factor=my_alpha,
    known_factors=ff_factors
)

print(corr_matrix)
#           Alpha  Mkt-RF    SMB    HML     RF
# Alpha      1.0   0.234  0.120  0.089  0.001
# Mkt-RF     0.234  1.0   0.123  0.456  0.012
# ...
```

#### Full Analysis Report:

```python
# Comprehensive report
report = analyzer.factor_analysis_report(
    alpha_factor=my_alpha,
    factor_returns=ff_factors,
    alpha_name="RL_Generated_Alpha_001"
)

analyzer.print_factor_report(report)
```

---

### 3. EvaluationValidator (`validator.py`)

Validates backtest results for statistical significance and robustness.

#### Key Features:
- **IC Significance**: T-test that IC is significantly different from 0
- **Sharpe Ratio Test**: Verify Sharpe exceeds threshold
- **Drawdown Test**: Maximum drawdown doesn't exceed limits
- **Turnover Test**: Check transaction costs aren't too high
- **Consistency Test**: Performance stable across rolling windows
- **Overfitting Test**: Compare in-sample vs out-of-sample performance
- **Walk-Forward Analysis**: Multi-period robustness assessment

#### Usage:

```python
from src.evaluation.validator import EvaluationValidator

validator = EvaluationValidator()

# IC Significance Test
ic_test = validator.ic_significance_test(
    ic_series=ic_values,
    min_samples=30
)
print(f"IC Significant at 95%: {ic_test['significant_95']}")
print(f"IC Mean: {ic_test['mean_ic']:.4f}, p-value: {ic_test['p_value']:.4f}")

# Sharpe Ratio Test
sharpe_test = validator.sharpe_ratio_test(
    returns=portfolio_returns,
    target_sharpe=1.0,
    periods_per_year=252
)
print(f"Sharpe {sharpe_test['sharpe_ratio']:.2f} >= Target {sharpe_test['target']}")

# Maximum Drawdown Test
dd_test = validator.maximum_drawdown_test(
    returns=portfolio_returns,
    max_dd_threshold=-0.20  # Maximum -20% drawdown
)
print(f"Max DD {dd_test['max_drawdown']:.2%} >= Threshold {dd_test['threshold']:.2%}")

# Turnover Test
turnover_test = validator.turnover_test(
    turnover=turnover_series,
    max_turnover=0.50  # Maximum 50% turnover
)
print(f"Avg Turnover {turnover_test['avg_turnover']:.2%} <= Max {turnover_test['threshold']:.2%}")
```

#### Consistency Test:

```python
# Check if returns are stable across time
consistency = validator.consistency_test(
    returns=portfolio_returns,
    window=60  # 60-day rolling windows
)
print(f"Consistency Score: {consistency['consistency_score']:.2f}")
print(f"Consistent Factor: {consistency['passed']}")
```

#### Overfitting Detection:

```python
# Compare in-sample vs out-of-sample Sharpe
overfitting_test = validator.overfitting_test(
    in_sample_sharpe=2.0,
    out_of_sample_sharpe=1.5,
    degradation_threshold=0.30  # Allow 30% degradation
)
print(f"Overfitted: {overfitting_test['overfitted']}")
print(f"Degradation: {overfitting_test['degradation']:.2%}")
```

#### Walk-Forward Analysis:

```python
# Multi-period robustness test
wf_results = validator.walk_forward_analysis(
    returns=returns_df,
    train_window=252,    # 1 year training
    test_window=63       # 3 months testing
)
print(f"Periods Tested: {wf_results['num_periods']}")
print(f"Mean Train Sharpe: {wf_results['mean_train_sharpe']:.2f}")
print(f"Mean Test Sharpe: {wf_results['mean_test_sharpe']:.2f}")
print(f"Robust: {wf_results['robust']}")
```

#### Full Validation Report:

```python
# Run all tests
report = validator.full_validation(
    returns=portfolio_returns,
    ic_series=ic_values,
    turnover=turnover_series,
    factor_alpha=0.01
)

# Pretty print
validator.print_validation_report(report)
```

---

## Integration with RL Pipeline

### Typical Evaluation Workflow:

```python
from src.evaluation import CrossSectionalBacktester, FactorAnalyzer
from src.evaluation.validator import EvaluationValidator

# 1. GENERATE FACTOR from RL agent
# (In practice, this comes from your RL environment)
rl_factor = generate_rl_factor()
forward_returns = get_returns()

# 2. BACKTEST
backtester = CrossSectionalBacktester(periods_per_year=252)
quintile_results = backtester.quintile_sort(
    factors=rl_factor,
    returns=forward_returns,
    lag=1
)
ic_series = backtester.information_coefficient(rl_factor, forward_returns)

print(f"L/S Sharpe: {quintile_results['LS_Spread']['sharpe']:.2f}")
print(f"Mean IC: {ic_series.mean():.4f}")

# 3. FACTOR ANALYSIS
analyzer = FactorAnalyzer()
ff_factors = analyzer.fetch_fama_french_factors()
analysis = analyzer.factor_analysis_report(
    alpha_factor=rl_factor.mean(axis=1),  # Convert to single series
    factor_returns=ff_factors
)

print(f"Alpha: {analysis['regression']['alpha']:.4%}")
print(f"Idiosyncratic: {analysis['variance_decomposition']['idiosyncratic_pct']:.1f}%")

# 4. VALIDATE
validator = EvaluationValidator()
report = validator.full_validation(
    returns=quintile_results['LS_Spread'],
    ic_series=ic_series,
    turnover=turnover_series
)

if report['overall_passed']:
    print("✓ Factor passed all validation checks!")
    save_factor_to_production(rl_factor)
else:
    print("✗ Factor failed validation, continue training...")
```

---

## Key Metrics Reference

### Information Coefficient (IC)
- **Definition**: Rank correlation between factor and forward returns
- **Interpretation**: 
  - IC > 0.05: Good predictive power
  - IC > 0.10: Excellent predictive power
  - IC < 0.02: Weak signal

### Sharpe Ratio
- **Definition**: (Return - Risk-Free Rate) / Volatility
- **Interpretation**:
  - Sharpe > 1.0: Good risk-adjusted returns
  - Sharpe > 2.0: Excellent
  - Sharpe < 0.5: Poor

### Maximum Drawdown
- **Definition**: Largest peak-to-trough decline
- **Interpretation**:
  - DD < -10%: Acceptable
  - DD < -20%: Concerning
  - DD < -50%: High risk

### Turnover
- **Definition**: Sum of absolute weight changes / 2
- **Interpretation**:
  - TO < 10%: Low turnover
  - TO > 50%: High costs
  - TO > 100%: Excessive

### Alpha (Fama-French)
- **Definition**: Excess return after accounting for systematic factors
- **Interpretation**:
  - Alpha > 0: True excess returns
  - Alpha > 0.5% (annualized): Significant
  - Alpha < 0: Underperformance

---

## Performance Thresholds for RL Alpha

For production-ready alpha factors:

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| IC Mean | 0.02 | 0.05 | 0.10 |
| IC t-stat | 1.96 | 2.58 | 3.29 |
| L/S Sharpe | 0.5 | 1.0 | 2.0 |
| Max DD | -30% | -15% | -10% |
| Turnover | N/A | <30% | <15% |
| Alpha (ann) | 0.0% | 0.5% | 1.0% |
| Idiosyncratic % | 50% | 75% | 90% |

---

## Examples

See `src/evaluation/example.py` for:
- Quintile backtest analysis
- Turnover and transaction cost analysis
- Fama-French regression
- Factor orthogonalization
- Validation and robustness checks
- Full evaluation pipeline

Run examples:
```bash
cd src/evaluation
python example.py
```

---

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scipy`: Statistical functions
- `pandas-datareader`: Fetch Fama-French factors

---

## Future Enhancements

- [ ] Support for transaction cost modeling
- [ ] Multi-period attribution analysis
- [ ] Machine learning-based overfitting detection
- [ ] Real-time performance monitoring
- [ ] Integration with portfolio optimization
