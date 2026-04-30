"""
Cross-sectional backtester for RL-generated alpha factors.

Provides:
- Long-short quintile/decile backtests
- Information Coefficient (IC) analysis
- Turnover metrics
- Performance attribution
- Out-of-sample validation
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class CrossSectionalBacktester:
    """
    Backtester for cross-sectional alpha factors.
    
    Implements:
    - Quintile/Decile sorting
    - Long-short portfolio construction
    - Information Coefficient (rank and regular)
    - Turnover calculations
    - Return analysis (mean, std, Sharpe)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ):
        """
        Initialize backtester.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio (default 2%)
            periods_per_year: Trading periods per year (252 for daily, 252*6.5 for intraday)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.daily_risk_free = risk_free_rate / periods_per_year

    def rank_correlation_ic(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        lag: int = 1,
    ) -> pd.Series:
        """
        Calculate Rank Information Coefficient (Spearman correlation).
        
        Measures predictive power of factor for forward returns.
        
        Args:
            factors: Cross-sectional factor values (assets × time)
            returns: Forward returns (assets × time)
            lag: Forward-looking lag for returns (1 = next period)
            
        Returns:
            Series of rank IC values over time
        """
        if factors.shape != returns.shape:
            raise ValueError("Factors and returns must have same shape")
        
        ic_values = []
        for t in range(len(factors) - lag):
            factor_cs = factors.iloc[t]
            return_cs = returns.iloc[t + lag]
            
            # Remove NaN values
            valid_idx = factor_cs.notna() & return_cs.notna()
            if valid_idx.sum() < 3:
                ic_values.append(np.nan)
                continue
            
            # Spearman rank correlation
            ic = stats.spearmanr(factor_cs[valid_idx], return_cs[valid_idx])[0]
            ic_values.append(ic)
        
        return pd.Series(ic_values, index=factors.index[:-lag])

    def information_coefficient(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        lag: int = 1,
        method: str = "rank",
    ) -> pd.Series:
        """
        Calculate Information Coefficient.
        
        Args:
            factors: Cross-sectional factor values
            returns: Forward returns
            lag: Forward-looking lag
            method: 'rank' (Spearman) or 'pearson'
            
        Returns:
            Series of IC values
        """
        if method == "rank":
            return self.rank_correlation_ic(factors, returns, lag)
        
        elif method == "pearson":
            ic_values = []
            for t in range(len(factors) - lag):
                factor_cs = factors.iloc[t]
                return_cs = returns.iloc[t + lag]
                
                valid_idx = factor_cs.notna() & return_cs.notna()
                if valid_idx.sum() < 3:
                    ic_values.append(np.nan)
                    continue
                
                ic = stats.pearsonr(factor_cs[valid_idx], return_cs[valid_idx])[0]
                ic_values.append(ic)
            
            return pd.Series(ic_values, index=factors.index[:-lag])
        
        else:
            raise ValueError(f"Unknown method: {method}")

    def quintile_sort(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        lag: int = 1,
    ) -> Dict[str, object]:
        """
        Quintile sort backtest (long top quintile, short bottom).
        
        Args:
            factors: Factor values (assets × time)
            returns: Forward returns (assets × time)
            lag: Forward-looking lag
            
        Returns:
            Dictionary with quintile analysis results
        """
        returns_shifted = returns.shift(-lag).dropna()
        factors_aligned = factors.loc[returns_shifted.index]
        
        quintile_returns = {}
        quintile_holdings = {}
        
        for t in range(len(factors_aligned)):
            factor_cross = factors_aligned.iloc[t]
            return_cross = returns_shifted.iloc[t]
            
            # Skip if insufficient data
            if factor_cross.isna().sum() > len(factor_cross) * 0.5:
                continue
            
            # Assign quintiles
            quintiles = pd.qcut(factor_cross, q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
            
            # Calculate returns per quintile
            for q in range(1, 6):
                if q not in quintile_returns:
                    quintile_returns[q] = []
                    quintile_holdings[q] = []
                
                members = quintiles == q
                if members.sum() > 0:
                    q_return = return_cross[members].mean()
                    quintile_returns[q].append(q_return)
                    quintile_holdings[q].append(members.sum())
                else:
                    quintile_returns[q].append(np.nan)
                    quintile_holdings[q].append(0)
        
        # Aggregate results
        results = {}
        for q in range(1, 6):
            returns_series = pd.Series(quintile_returns[q])
            results[f"Q{q}"] = {
                "mean_return": returns_series.mean(),
                "std_return": returns_series.std(),
                "sharpe": self._calculate_sharpe(returns_series),
                "total_return": (1 + returns_series).prod() - 1,
                "avg_holdings": np.mean(quintile_holdings[q]),
            }
        
        # Long-short spread
        long_short = pd.Series(quintile_returns[5]) - pd.Series(quintile_returns[1])
        results["LS_Spread"] = {
            "mean_return": long_short.mean(),
            "std_return": long_short.std(),
            "sharpe": self._calculate_sharpe(long_short),
            "total_return": (1 + long_short).prod() - 1,
            "t_stat": stats.ttest_1sample(long_short.dropna(), 0)[0],
        }
        
        return results

    def decile_sort(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        lag: int = 1,
    ) -> Dict[str, object]:
        """
        Decile sort backtest (10 buckets, long top vs short bottom).
        
        Args:
            factors: Factor values
            returns: Forward returns
            lag: Forward-looking lag
            
        Returns:
            Dictionary with decile analysis results
        """
        returns_shifted = returns.shift(-lag).dropna()
        factors_aligned = factors.loc[returns_shifted.index]
        
        decile_returns = {}
        
        for t in range(len(factors_aligned)):
            factor_cross = factors_aligned.iloc[t]
            return_cross = returns_shifted.iloc[t]
            
            if factor_cross.isna().sum() > len(factor_cross) * 0.5:
                continue
            
            # Assign deciles
            deciles = pd.qcut(factor_cross, q=10, labels=range(1, 11), duplicates="drop")
            
            # Returns per decile
            for d in range(1, 11):
                if d not in decile_returns:
                    decile_returns[d] = []
                
                members = deciles == d
                if members.sum() > 0:
                    d_return = return_cross[members].mean()
                    decile_returns[d].append(d_return)
                else:
                    decile_returns[d].append(np.nan)
        
        # Aggregate
        results = {}
        for d in range(1, 11):
            returns_series = pd.Series(decile_returns[d])
            results[f"D{d}"] = {
                "mean_return": returns_series.mean(),
                "total_return": (1 + returns_series).prod() - 1,
            }
        
        # Long-short
        long_short = pd.Series(decile_returns[10]) - pd.Series(decile_returns[1])
        results["LS_Spread"] = {
            "mean_return": long_short.mean(),
            "sharpe": self._calculate_sharpe(long_short),
            "t_stat": stats.ttest_1sample(long_short.dropna(), 0)[0],
        }
        
        return results

    def portfolio_returns(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate portfolio returns from weights and returns.
        
        Args:
            weights: Portfolio weights (assets × time, normalized)
            returns: Asset returns (assets × time)
            
        Returns:
            Portfolio returns over time
        """
        # Element-wise multiplication then sum
        daily_returns = (weights * returns).sum(axis=0)
        return daily_returns

    def calculate_turnover(
        self,
        weights_t0: pd.Series,
        weights_t1: pd.Series,
    ) -> float:
        """
        Calculate portfolio turnover between two periods.
        
        Turnover = sum of absolute weight changes / 2
        
        Args:
            weights_t0: Portfolio weights at time t
            weights_t1: Portfolio weights at time t+1
            
        Returns:
            Turnover ratio (0 to 1)
        """
        weight_changes = (weights_t1 - weights_t0).abs().sum() / 2
        return weight_changes

    def cumulative_turnover(
        self,
        weights: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate cumulative turnover over time.
        
        Args:
            weights: Portfolio weights over time
            
        Returns:
            Series of cumulative turnover
        """
        turnover_series = []
        for t in range(1, len(weights)):
            to = self.calculate_turnover(weights.iloc[t - 1], weights.iloc[t])
            turnover_series.append(to)
        
        return pd.Series(turnover_series, index=weights.index[1:])

    def transaction_costs(
        self,
        turnover: pd.Series,
        bid_ask_spread: float = 0.001,  # 10 bps
    ) -> pd.Series:
        """
        Estimate transaction costs from turnover.
        
        Args:
            turnover: Portfolio turnover per period
            bid_ask_spread: Round-trip cost (default 10 bps)
            
        Returns:
            Series of transaction cost impacts
        """
        return turnover * bid_ask_spread

    def risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns for relative metrics
            
        Returns:
            Dictionary of risk metrics
        """
        returns_clean = returns.dropna()
        
        # Basic stats
        metrics = {
            "mean_return": returns_clean.mean() * self.periods_per_year,
            "std_return": returns_clean.std() * np.sqrt(self.periods_per_year),
            "sharpe_ratio": self._calculate_sharpe(returns_clean),
            "max_drawdown": self._calculate_max_drawdown(returns_clean),
            "calmar_ratio": (returns_clean.mean() * self.periods_per_year) / 
                           abs(self._calculate_max_drawdown(returns_clean)) if self._calculate_max_drawdown(returns_clean) != 0 else np.nan,
            "skewness": stats.skew(returns_clean),
            "kurtosis": stats.kurtosis(returns_clean),
            "win_rate": (returns_clean > 0).sum() / len(returns_clean),
        }
        
        # Relative to benchmark
        if benchmark_returns is not None:
            benchmark_clean = benchmark_returns.loc[returns_clean.index]
            excess = returns_clean - benchmark_clean
            metrics["alpha"] = excess.mean() * self.periods_per_year
            metrics["beta"] = np.cov(returns_clean, benchmark_clean)[0, 1] / np.var(benchmark_clean)
            metrics["information_ratio"] = excess.mean() / excess.std() * np.sqrt(self.periods_per_year)
        
        return metrics

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        excess = returns - self.daily_risk_free
        if excess.std() == 0:
            return 0
        return excess.mean() / excess.std() * np.sqrt(self.periods_per_year)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def factor_performance_table(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        lag: int = 1,
    ) -> pd.DataFrame:
        """
        Create comprehensive factor performance summary table.
        
        Args:
            factors: Factor values
            returns: Forward returns
            lag: Forward-looking lag
            
        Returns:
            Performance summary DataFrame
        """
        # IC analysis
        ic_series = self.information_coefficient(factors, returns, lag, method="rank")
        
        # Quintile analysis
        quintile_results = self.quintile_sort(factors, returns, lag)
        
        # Summary table
        summary = pd.DataFrame({
            "IC_Mean": [ic_series.mean()],
            "IC_Std": [ic_series.std()],
            "IC_t-stat": [ic_series.mean() / ic_series.std() * np.sqrt(len(ic_series))],
            "Q1_Return": [quintile_results["Q1"]["mean_return"]],
            "Q5_Return": [quintile_results["Q5"]["mean_return"]],
            "LS_Return": [quintile_results["LS_Spread"]["mean_return"]],
            "LS_Sharpe": [quintile_results["LS_Spread"]["sharpe"]],
            "LS_t-stat": [quintile_results["LS_Spread"]["t_stat"]],
        })
        
        return summary

    def plot_quintile_returns(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        lag: int = 1,
    ) -> Dict[str, float]:
        """
        Calculate returnsper quintile for plotting.
        
        Args:
            factors: Factor values
            returns: Forward returns
            lag: Forward-looking lag
            
        Returns:
            Dictionary mapping quintile → average return
        """
        results = self.quintile_sort(factors, returns, lag)
        
        plot_data = {}
        for q in range(1, 6):
            plot_data[f"Q{q}"] = results[f"Q{q}"]["mean_return"]
        plot_data["LS"] = results["LS_Spread"]["mean_return"]
        
        return plot_data
