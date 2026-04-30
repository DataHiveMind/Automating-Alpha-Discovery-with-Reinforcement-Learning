"""
Data validation module for quality assurance and error detection.

Provides comprehensive checks for data integrity, completeness, and market sanity.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class DataValidator:
    """
    Validates market data quality and completeness.
    
    Checks for:
    - Missing values
    - Price anomalies (gaps, reversals)
    - Volume anomalies
    - Data consistency across assets
    """

    @staticmethod
    def check_missing_data(data: pd.DataFrame, threshold: float = 0.1) -> Dict[str, object]:
        """
        Check for missing data.
        
        Args:
            data: DataFrame to validate
            threshold: Acceptable missing data ratio (default 10%)
            
        Returns:
            Dictionary with missing data report
        """
        missing_pct = (data.isna().sum() / len(data) * 100).round(2)
        violations = missing_pct[missing_pct > threshold * 100]
        
        return {
            "total_missing": data.isna().sum().sum(),
            "missing_pct": missing_pct.to_dict(),
            "violations": violations.to_dict() if len(violations) > 0 else {},
            "passed": len(violations) == 0,
        }

    @staticmethod
    def check_price_gaps(
        data: pd.DataFrame,
        price_col: str = "Close",
        gap_threshold: float = 0.20,
    ) -> Dict[str, object]:
        """
        Detect suspicious price gaps (>20% overnight movements).
        
        Args:
            data: OHLCV data
            price_col: Price column to check
            gap_threshold: Threshold for gap detection (default 20%)
            
        Returns:
            Dictionary with gap detection report
        """
        if price_col not in data.columns:
            return {"error": f"Column {price_col} not found"}
        
        gaps = (data[price_col] / data[price_col].shift(1) - 1).abs()
        suspicious = gaps[gaps > gap_threshold]
        
        return {
            "total_gaps_detected": len(suspicious),
            "max_gap": gaps.max() * 100,
            "suspicious_dates": suspicious.index.tolist() if len(suspicious) > 0 else [],
            "passed": len(suspicious) < len(data) * 0.05,  # Allow <5% suspicious gaps
        }

    @staticmethod
    def check_volume_anomalies(
        data: pd.DataFrame,
        volume_col: str = "Volume",
        std_threshold: float = 3.0,
    ) -> Dict[str, object]:
        """
        Detect volume anomalies using z-score method.
        
        Args:
            data: OHLCV data
            volume_col: Volume column to check
            std_threshold: Z-score threshold (default 3 std)
            
        Returns:
            Dictionary with anomaly report
        """
        if volume_col not in data.columns:
            return {"error": f"Column {volume_col} not found"}
        
        volumes = data[volume_col].fillna(0)
        mean_vol = volumes.mean()
        std_vol = volumes.std()
        
        z_scores = (volumes - mean_vol) / (std_vol + 1e-8)
        anomalies = np.abs(z_scores) > std_threshold
        
        return {
            "total_anomalies": anomalies.sum(),
            "mean_volume": round(mean_vol, 0),
            "std_volume": round(std_vol, 0),
            "anomaly_pct": round(anomalies.sum() / len(data) * 100, 2),
            "passed": anomalies.sum() / len(data) < 0.05,  # <5% anomalies acceptable
        }

    @staticmethod
    def check_ohlc_consistency(data: pd.DataFrame) -> Dict[str, object]:
        """
        Verify OHLC (Open, High, Low, Close) consistency.
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary with consistency report
        """
        required = ["Open", "High", "Low", "Close"]
        missing = [col for col in required if col not in data.columns]
        
        if missing:
            return {"error": f"Missing columns: {missing}"}
        
        violations = (
            (data["High"] < data["Low"]) |
            (data["High"] < data["Open"]) |
            (data["High"] < data["Close"]) |
            (data["Low"] > data["Open"]) |
            (data["Low"] > data["Close"])
        )
        
        return {
            "total_violations": violations.sum(),
            "violation_pct": round(violations.sum() / len(data) * 100, 2),
            "passed": violations.sum() == 0,
        }

    @staticmethod
    def check_returns_distribution(
        data: pd.DataFrame,
        price_col: str = "Close",
        skew_threshold: float = 2.0,
    ) -> Dict[str, object]:
        """
        Check returns distribution for extreme skewness.
        
        Args:
            data: OHLCV data
            price_col: Price column
            skew_threshold: Acceptable skewness threshold
            
        Returns:
            Dictionary with distribution report
        """
        if price_col not in data.columns:
            return {"error": f"Column {price_col} not found"}
        
        returns = data[price_col].pct_change().dropna()
        
        from scipy import stats
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            "mean_return": round(returns.mean() * 10000, 2),  # bps
            "std_return": round(returns.std() * 10000, 2),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "passed": abs(skewness) < skew_threshold,
        }

    @staticmethod
    def full_validation(data: pd.DataFrame, price_col: str = "Close") -> Dict[str, object]:
        """
        Run comprehensive validation on data.
        
        Args:
            data: OHLCV data
            price_col: Price column name
            
        Returns:
            Complete validation report
        """
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_rows": len(data),
            "date_range": {
                "start": data.index[0].isoformat() if len(data) > 0 else None,
                "end": data.index[-1].isoformat() if len(data) > 0 else None,
            },
            "missing_data": DataValidator.check_missing_data(data),
            "price_gaps": DataValidator.check_price_gaps(data, price_col),
            "volume_anomalies": DataValidator.check_volume_anomalies(data),
            "ohlc_consistency": DataValidator.check_ohlc_consistency(data),
            "returns_distribution": DataValidator.check_returns_distribution(data, price_col),
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
        print("\n" + "="*60)
        print("DATA VALIDATION REPORT")
        print("="*60)
        
        print(f"\nTimestamp: {report['timestamp']}")
        print(f"Total Rows: {report['total_rows']}")
        print(f"Date Range: {report['date_range']['start']} to {report['date_range']['end']}")
        
        print(f"\n{'Check':<25} {'Result':<10}")
        print("-" * 35)
        
        for key, value in report.items():
            if isinstance(value, dict) and "passed" in value:
                status = "✓ PASS" if value["passed"] else "✗ FAIL"
                print(f"{key:<25} {status:<10}")
        
        overall = "✓ ALL PASSED" if report["overall_passed"] else "✗ SOME FAILED"
        print("\n" + "="*60)
        print(f"OVERALL: {overall}")
        print("="*60 + "\n")
