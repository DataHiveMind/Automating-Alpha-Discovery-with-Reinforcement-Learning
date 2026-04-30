"""
Example and integration script for the evaluation module.

Demonstrates:
- Cross-sectional backtesting
- Factor analysis and Fama-French regression
- Validation and robustness checks
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backtester import CrossSectionalBacktester
from factor_analysis import FactorAnalyzer
from validator import EvaluationValidator


def example_basic_backtest():
    """Basic quintile backtest example."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Cross-Sectional Quintile Backtest")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_assets = 100
    n_periods = 252  # 1 year of daily data
    
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq="D")
    
    # Factor: random quality factor with some predictive power
    factors = pd.DataFrame(
        np.random.randn(n_periods, n_assets),
        index=dates,
        columns=[f"Asset_{i}" for i in range(n_assets)]
    )
    
    # Returns: partially driven by factor + noise
    factor_contrib = factors * 0.03  # 3% factor contribution
    noise = np.random.randn(n_periods, n_assets) * 0.02
    returns = factor_contrib + noise
    
    # Run backtest
    backtester = CrossSectionalBacktester(periods_per_year=252)
    
    # Quintile analysis
    quintile_results = backtester.quintile_sort(factors, returns, lag=1)
    
    print("\nQuintile Returns:")
    for q in range(1, 6):
        q_data = quintile_results[f"Q{q}"]
        print(f"  Q{q}: Mean={q_data['mean_return']:.4f}, "
              f"Sharpe={q_data['sharpe']:.2f}, "
              f"Total={q_data['total_return']:.2%}")
    
    ls = quintile_results["LS_Spread"]
    print(f"\n  Long-Short: Mean={ls['mean_return']:.4f}, "
          f"Sharpe={ls['sharpe']:.2f}, "
          f"t-stat={ls['t_stat']:.2f}")
    
    # IC analysis
    ic_series = backtester.information_coefficient(factors, returns, lag=1)
    print(f"\nInformation Coefficient:")
    print(f"  Mean IC: {ic_series.mean():.4f}")
    print(f"  Std IC:  {ic_series.std():.4f}")
    print(f"  Hit Rate: {(ic_series > 0).sum() / len(ic_series):.2%}")


def example_turnover_analysis():
    """Turnover and transaction cost analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Turnover and Transaction Cost Analysis")
    print("="*70)
    
    np.random.seed(42)
    n_assets = 100
    n_periods = 100
    
    # Generate random portfolio weights
    weights = pd.DataFrame(
        np.random.dirichlet(np.ones(n_assets), n_periods),
        columns=[f"Asset_{i}" for i in range(n_assets)]
    )
    
    backtester = CrossSectionalBacktester()
    
    # Calculate turnover
    turnover_series = backtester.cumulative_turnover(weights)
    
    print(f"\nTurnover Statistics:")
    print(f"  Mean: {turnover_series.mean():.2%}")
    print(f"  Std:  {turnover_series.std():.2%}")
    print(f"  Max:  {turnover_series.max():.2%}")
    print(f"  Min:  {turnover_series.min():.2%}")
    
    # Transaction costs (assuming 10 bps round-trip)
    costs = backtester.transaction_costs(turnover_series, bid_ask_spread=0.001)
    
    print(f"\nTransaction Costs (10 bps round-trip):")
    print(f"  Mean: {costs.mean():.4f}")
    print(f"  Total: {costs.sum():.4f}")


def example_factor_analysis():
    """Fama-French regression and factor exposure."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Fama-French Factor Analysis")
    print("="*70)
    
    # Generate synthetic factor returns and asset returns
    np.random.seed(42)
    n_periods = 252
    
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq="D")
    
    # Synthetic Fama-French factors
    factors = pd.DataFrame({
        "Mkt-RF": np.random.normal(0.0004, 0.01, n_periods),
        "SMB": np.random.normal(0.0001, 0.005, n_periods),
        "HML": np.random.normal(0.0001, 0.005, n_periods),
        "RF": np.full(n_periods, 0.00002),
    }, index=dates)
    
    # Synthetic alpha factor (exposed to market and SMB, orthogonal to HML)
    alpha_factor = (
        0.0005 +  # Alpha
        0.8 * factors["Mkt-RF"] +  # Market exposure
        0.3 * factors["SMB"] +  # Size exposure
        np.random.normal(0, 0.005, n_periods)  # Idiosyncratic
    )
    alpha_factor = pd.Series(alpha_factor, index=dates)
    
    # Run factor analysis
    analyzer = FactorAnalyzer()
    
    reg_results = analyzer.fama_french_regression(alpha_factor, factors)
    
    print("\nFama-French 3-Factor Regression:")
    print(f"  Alpha (ann):    {reg_results['alpha_annualized']:.4%}")
    print(f"  Alpha t-stat:   {reg_results['t_stats']['Alpha']:.2f}")
    print(f"  Mkt-RF Beta:    {reg_results['coefficients']['Mkt-RF']:.4f}")
    print(f"  SMB Beta:       {reg_results['coefficients']['SMB']:.4f}")
    print(f"  HML Beta:       {reg_results['coefficients']['HML']:.4f}")
    print(f"  R-squared:      {reg_results['r_squared']:.4f}")
    
    # Variance decomposition
    var_decomp = analyzer.idiosyncratic_variance(alpha_factor, factors)
    
    print(f"\nVariance Decomposition:")
    print(f"  Total Var:      {var_decomp['total_variance']:.6f}")
    print(f"  Explained:      {var_decomp['explained_variance']:.6f} ({var_decomp['explained_variance']/var_decomp['total_variance']*100:.1f}%)")
    print(f"  Idiosyncratic:  {var_decomp['idiosyncratic_variance']:.6f} ({var_decomp['idiosyncratic_pct']:.1f}%)")


def example_orthogonalization():
    """Orthogonalize alpha against known factors."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Factor Orthogonalization")
    print("="*70)
    
    np.random.seed(42)
    n_periods = 252
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq="D")
    
    # Known factors
    factors = pd.DataFrame({
        "Mkt-RF": np.random.normal(0.0004, 0.01, n_periods),
        "SMB": np.random.normal(0.0001, 0.005, n_periods),
        "HML": np.random.normal(0.0001, 0.005, n_periods),
    }, index=dates)
    
    # Alpha factor (exposed to all factors)
    alpha_factor = (
        0.0005 + 0.8 * factors["Mkt-RF"] + 0.3 * factors["SMB"] + 
        0.2 * factors["HML"] + np.random.normal(0, 0.005, n_periods)
    )
    alpha_factor = pd.Series(alpha_factor, index=dates)
    
    analyzer = FactorAnalyzer()
    
    print(f"Original Factor Correlations:")
    corr_orig = alpha_factor.corr(factors["Mkt-RF"])
    print(f"  Corr with Mkt-RF: {corr_orig:.4f}")
    
    # Orthogonalize
    ortho = analyzer.orthogonalize_factor(alpha_factor, factors[["Mkt-RF", "SMB", "HML"]])
    
    print(f"\nOrthogonalized Factor Correlations:")
    corr_ortho = ortho.corr(factors["Mkt-RF"])
    print(f"  Corr with Mkt-RF: {corr_ortho:.4f} (reduced)")
    
    print(f"\nOrthogonalized Alpha Std: {ortho.std():.6f}")
    print(f"Original Alpha Std:       {alpha_factor.std():.6f}")


def example_validation():
    """Validation and robustness checks."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Backtest Validation")
    print("="*70)
    
    np.random.seed(42)
    n_periods = 252
    
    # Generate returns with positive Sharpe
    returns = pd.Series(
        0.0005 + np.random.normal(0, 0.01, n_periods),
        index=pd.date_range(end=datetime.now(), periods=n_periods, freq="D")
    )
    
    ic_series = pd.Series(
        np.random.uniform(-0.1, 0.2, n_periods),
        index=returns.index
    )
    
    turnover = pd.Series(
        np.random.uniform(0.05, 0.30, n_periods),
        index=returns.index
    )
    
    validator = EvaluationValidator()
    
    # Individual tests
    ic_test = validator.ic_significance_test(ic_series)
    print(f"\nIC Significance Test:")
    print(f"  Mean IC: {ic_test['mean_ic']:.4f}")
    print(f"  t-stat:  {ic_test['t_stat']:.2f}")
    print(f"  p-value: {ic_test['p_value']:.4f}")
    print(f"  Passed: {ic_test['passed']}")
    
    sharpe_test = validator.sharpe_ratio_test(returns, target_sharpe=0.5)
    print(f"\nSharpe Ratio Test (target=0.5):")
    print(f"  Sharpe:  {sharpe_test['sharpe_ratio']:.2f}")
    print(f"  Passed: {sharpe_test['passed']}")
    
    dd_test = validator.maximum_drawdown_test(returns)
    print(f"\nMax Drawdown Test:")
    print(f"  Max DD: {dd_test['max_drawdown']:.2%}")
    print(f"  Passed: {dd_test['passed']}")
    
    to_test = validator.turnover_test(turnover)
    print(f"\nTurnover Test:")
    print(f"  Avg Turnover: {to_test['avg_turnover']:.2%}")
    print(f"  Passed: {to_test['passed']}")
    
    # Full validation
    print(f"\n" + "-"*70)
    report = validator.full_validation(returns, ic_series, turnover)
    validator.print_validation_report(report)


def example_comprehensive_pipeline():
    """Full evaluation pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Comprehensive Evaluation Pipeline")
    print("="*70)
    
    # Generate realistic data
    np.random.seed(42)
    n_assets = 50
    n_periods = 500
    
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq="D")
    
    # Factor with predictive power
    factors = pd.DataFrame(
        0.1 * np.random.randn(n_periods, n_assets) +
        0.02 * np.random.randn(1, n_assets),  # Common signal
        index=dates,
        columns=[f"Asset_{i}" for i in range(n_assets)]
    )
    
    returns = 0.02 * factors + np.random.randn(n_periods, n_assets) * 0.02
    
    # Fama-French factors
    ff_factors = pd.DataFrame({
        "Mkt-RF": np.random.normal(0.0004, 0.01, n_periods),
        "SMB": np.random.normal(0.0001, 0.005, n_periods),
        "HML": np.random.normal(0.0001, 0.005, n_periods),
        "RF": np.full(n_periods, 0.00002),
    }, index=dates)
    
    # Run backtest
    print("\n[STEP 1: BACKTESTING]")
    backtester = CrossSectionalBacktester(periods_per_year=252)
    
    quintile_results = backtester.quintile_sort(factors, returns, lag=1)
    ic_series = backtester.information_coefficient(factors, returns, lag=1)
    
    ls_return = quintile_results["LS_Spread"]["mean_return"]
    print(f"  Long-Short Return: {ls_return:.4f}")
    print(f"  IC Mean: {ic_series.mean():.4f}")
    
    # Factor analysis
    print("\n[STEP 2: FACTOR ANALYSIS]")
    analyzer = FactorAnalyzer()
    
    # Create alpha factor from quintile spread
    alpha_factor = returns.mean(axis=1) * 0.1 + np.random.normal(0, 0.003, n_periods)
    alpha_factor = pd.Series(alpha_factor, index=dates)
    
    ff_reg = analyzer.fama_french_regression(alpha_factor, ff_factors)
    var_decomp = analyzer.idiosyncratic_variance(alpha_factor, ff_factors)
    
    print(f"  Alpha (ann): {ff_reg['alpha_annualized']:.4%}")
    print(f"  R-squared: {ff_reg['r_squared']:.4f}")
    print(f"  Idiosyncratic Var: {var_decomp['idiosyncratic_pct']:.1f}%")
    
    # Validation
    print("\n[STEP 3: VALIDATION]")
    validator = EvaluationValidator()
    
    ls_returns_series = returns.mean(axis=1) * 0.1
    turnover = pd.Series(np.random.uniform(0.1, 0.3, n_periods), index=dates)
    
    report = validator.full_validation(ls_returns_series, ic_series, turnover)
    
    print(f"  IC Significant: {report['ic_significance']['passed']}")
    print(f"  Sharpe Adequate: {report['sharpe_ratio']['passed']}")
    print(f"  Drawdown OK: {report['max_drawdown']['passed']}")
    print(f"  Turnover OK: {report['turnover']['passed']}")
    print(f"  Overall Valid: {report['overall_passed']}")
    
    print("\n" + "="*70)
    print("Pipeline Complete!")


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█  Evaluation Module Examples & Integration               █")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        example_basic_backtest()
        example_turnover_analysis()
        example_factor_analysis()
        example_orthogonalization()
        example_validation()
        example_comprehensive_pipeline()
        
        print("\n✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
