"""
Validation module for backtest and factor analysis results.

Provides quality checks for:
- Backtest robustness
- Statistical significance
- Out-of-sample validity
- Overfitting detection
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class EvaluationValidator:
    """
    Validates backtest and factor analysis results.
    
    Checks for:
    - Statistical significance
    - Overfitting
    - Out-of-sample robustness
    - Result stability
    """

    @staticmethod
    def ic_significance_test(
        ic_series: pd.Series,
        min_samples: int = 30,
    ) -> Dict[str, object]:
        """
        Test Information Coefficient for statistical significance.
        
        Args:
            ic_series: Series of IC values over time
            min_samples: Minimum samples required
            
        Returns:
            Significance test results
        """
        ic_clean = ic_series.dropna()
        
        if len(ic_clean) < min_samples:
            return {
                "passed": False,
                "reason": f"Insufficient samples: {len(ic_clean)} < {min_samples}",
            }
        
        # t-test for IC mean significantly different from 0
        t_stat, p_value = stats.ttest_1sample(ic_clean, 0)
        
        return {
            "passed": p_value < 0.05,
            "mean_ic": ic_clean.mean(),
            "std_ic": ic_clean.std(),
            "t_stat": t_stat,
            "p_value": p_value,
            "samples": len(ic_clean),
            "significant_95": p_value < 0.05,
            "significant_99": p_value < 0.01,
        }

    @staticmethod
    def sharpe_ratio_test(
        returns: pd.Series,
        target_sharpe: float = 1.0,
        periods_per_year: int = 252,
    ) -> Dict[str, object]:
        """
        Test Sharpe ratio against target threshold.
        
        Args:
            returns: Portfolio returns
            target_sharpe: Minimum acceptable Sharpe ratio
            periods_per_year: Periods per year
            
        Returns:
            Sharpe ratio test results
        """
        returns_clean = returns.dropna()
        
        sharpe = (returns_clean.mean() / returns_clean.std() * 
                 np.sqrt(periods_per_year))
        
        return {
            "passed": sharpe >= target_sharpe,
            "sharpe_ratio": sharpe,
            "target": target_sharpe,
            "excess_sharpe": sharpe - target_sharpe,
        }

    @staticmethod
    def maximum_drawdown_test(
        returns: pd.Series,
        max_dd_threshold: float = -0.20,
    ) -> Dict[str, object]:
        """
        Test maximum drawdown doesn't exceed threshold.
        
        Args:
            returns: Portfolio returns
            max_dd_threshold: Maximum acceptable drawdown (default -20%)
            
        Returns:
            Drawdown test results
        """
        returns_clean = returns.dropna()
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            "passed": max_dd >= max_dd_threshold,
            "max_drawdown": max_dd,
            "threshold": max_dd_threshold,
            "excess_drawdown": max_dd - max_dd_threshold,
        }

    @staticmethod
    def turnover_test(
        turnover: pd.Series,
        max_turnover: float = 0.50,
    ) -> Dict[str, object]:
        """
        Test turnover doesn't exceed threshold.
        
        Args:
            turnover: Daily turnover series
            max_turnover: Maximum acceptable turnover
            
        Returns:
            Turnover test results
        """
        turnover_clean = turnover.dropna()
        avg_turnover = turnover_clean.mean()
        
        return {
            "passed": avg_turnover <= max_turnover,
            "avg_turnover": avg_turnover,
            "threshold": max_turnover,
            "excess_turnover": avg_turnover - max_turnover,
        }

    @staticmethod
    def consistency_test(
        returns: pd.Series,
        window: int = 60,
    ) -> Dict[str, object]:
        """
        Test consistency of factor returns over rolling windows.
        
        Checks if performance is stable across different periods.
        
        Args:
            returns: Portfolio returns
            window: Rolling window size
            
        Returns:
            Consistency test results
        """
        returns_clean = returns.dropna()
        
        if len(returns_clean) < window * 2:
            return {
                "passed": False,
                "reason": "Insufficient data for consistency test",
            }
        
        rolling_sharpes = []
        for i in range(len(returns_clean) - window):
            window_returns = returns_clean.iloc[i:i+window]
            sharpe = (window_returns.mean() / window_returns.std() * 
                     np.sqrt(252))
            rolling_sharpes.append(sharpe)
        
        rolling_sharpes = pd.Series(rolling_sharpes)
        consistency = 1 - rolling_sharpes.std() / rolling_sharpes.mean()
        
        return {
            "passed": consistency > 0.5,  # 50% consistency threshold
            "consistency_score": consistency,
            "mean_rolling_sharpe": rolling_sharpes.mean(),
            "std_rolling_sharpe": rolling_sharpes.std(),
        }

    @staticmethod
    def overfitting_test(
        in_sample_sharpe: float,
        out_of_sample_sharpe: float,
        degradation_threshold: float = 0.30,
    ) -> Dict[str, object]:
        """
        Test for overfitting by comparing in-sample vs out-of-sample performance.
        
        Args:
            in_sample_sharpe: Sharpe ratio on training data
            out_of_sample_sharpe: Sharpe ratio on test data
            degradation_threshold: Maximum acceptable performance drop
            
        Returns:
            Overfitting test results
        """
        if in_sample_sharpe <= 0:
            return {
                "passed": False,
                "reason": "In-sample Sharpe must be positive",
            }
        
        degradation = (in_sample_sharpe - out_of_sample_sharpe) / in_sample_sharpe
        
        return {
            "passed": degradation <= degradation_threshold,
            "in_sample_sharpe": in_sample_sharpe,
            "out_of_sample_sharpe": out_of_sample_sharpe,
            "degradation": degradation,
            "threshold": degradation_threshold,
            "overfitted": degradation > degradation_threshold,
        }

    @staticmethod
    def walk_forward_analysis(
        returns: pd.DataFrame,
        train_window: int = 252,
        test_window: int = 63,
    ) -> Dict[str, object]:
        """
        Walk-forward analysis to detect overfitting and measure robustness.
        
        Args:
            returns: Returns DataFrame (time × factors)
            train_window: Training window size
            test_window: Test window size
            
        Returns:
            Walk-forward analysis results
        """
        n_periods = len(returns)
        sharpes_oos = []
        
        for start in range(0, n_periods - train_window - test_window, test_window):
            train_period = returns.iloc[start:start+train_window]
            test_period = returns.iloc[start+train_window:start+train_window+test_window]
            
            # Train metrics (on training period)
            train_sharpe = (train_period.mean() / train_period.std() * np.sqrt(252))
            
            # Test metrics (on test period)
            test_sharpe = (test_period.mean() / test_period.std() * np.sqrt(252))
            
            sharpes_oos.append({
                "train_sharpe": train_sharpe,
                "test_sharpe": test_sharpe,
                "degradation": (train_sharpe - test_sharpe) / abs(train_sharpe) if train_sharpe != 0 else np.nan,
            })
        
        sharpes_df = pd.DataFrame(sharpes_oos)
        
        return {
            "num_periods": len(sharpes_oos),
            "mean_train_sharpe": sharpes_df["train_sharpe"].mean(),
            "mean_test_sharpe": sharpes_df["test_sharpe"].mean(),
            "mean_degradation": sharpes_df["degradation"].mean(),
            "std_degradation": sharpes_df["degradation"].std(),
            "robust": sharpes_df["degradation"].mean() < 0.30,
        }

    @staticmethod
    def full_validation(
        returns: pd.Series,
        ic_series: pd.Series,
        turnover: pd.Series,
        factor_alpha: float = 0.01,
    ) -> Dict[str, object]:
        """
        Run comprehensive validation on backtest results.
        
        Args:
            returns: Portfolio returns
            ic_series: Information Coefficient series
            turnover: Daily turnover series
            factor_alpha: Alpha from factor regression
            
        Returns:
            Complete validation report
        """
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "ic_significance": EvaluationValidator.ic_significance_test(ic_series),
            "sharpe_ratio": EvaluationValidator.sharpe_ratio_test(returns),
            "max_drawdown": EvaluationValidator.maximum_drawdown_test(returns),
            "turnover": EvaluationValidator.turnover_test(turnover),
            "consistency": EvaluationValidator.consistency_test(returns),
        }
        
        # Overall pass/fail
        all_passed = all(
            v.get("passed", True) 
            for k, v in report.items() 
            if isinstance(v, dict) and "passed" in v
        )
        report["overall_passed"] = all_passed
        
        return report

    @staticmethod
    def print_validation_report(report: Dict[str, object]) -> None:
        """Pretty print validation report."""
        print("\n" + "="*70)
        print("EVALUATION VALIDATION REPORT")
        print("="*70)
        
        print(f"\nTimestamp: {report['timestamp']}")
        
        print(f"\n{'Check':<30} {'Result':<15} {'Details':<25}")
        print("-" * 70)
        
        for key, value in report.items():
            if isinstance(value, dict) and "passed" in value:
                status = "✓ PASS" if value["passed"] else "✗ FAIL"
                
                # Extract main metric
                if key == "ic_significance":
                    detail = f"IC = {value['mean_ic']:.4f}"
                elif key == "sharpe_ratio":
                    detail = f"Sharpe = {value['sharpe_ratio']:.2f}"
                elif key == "max_drawdown":
                    detail = f"DD = {value['max_drawdown']:.2%}"
                elif key == "turnover":
                    detail = f"Turnover = {value['avg_turnover']:.2%}"
                elif key == "consistency":
                    detail = f"Consistency = {value['consistency_score']:.2f}"
                else:
                    detail = ""
                
                print(f"{key:<30} {status:<15} {detail:<25}")
        
        overall = "✓ ALL PASSED" if report["overall_passed"] else "✗ SOME FAILED"
        print("\n" + "="*70)
        print(f"OVERALL: {overall}")
        print("="*70 + "\n")
