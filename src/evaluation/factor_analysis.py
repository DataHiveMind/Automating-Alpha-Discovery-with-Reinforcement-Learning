"""
Factor analysis module for Fama-French regression and orthogonalization.

Provides:
- Factor loading calculations
- Fama-French 3-factor and 5-factor models
- Factor orthogonalization
- Risk attribution
- Correlation analysis
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class FactorAnalyzer:
    """
    Analyzes factor exposure to known risk factors.
    
    Implements:
    - Fama-French regression
    - Factor orthogonalization (Gram-Schmidt)
    - Correlation analysis
    - Risk decomposition
    - Alpha extraction
    """

    def __init__(self):
        """Initialize factor analyzer."""
        pass

    # ========== DATA LOADING ==========

    def fetch_fama_french_factors(
        self,
        freq: str = "daily",
        model: str = "3f",
    ) -> pd.DataFrame:
        """
        Fetch Fama-French factors from data source.
        
        Args:
            freq: 'daily', 'monthly', 'annual'
            model: '3f' (Mkt-RF, SMB, HML), '5f' (add RMW, CMA)
            
        Returns:
            DataFrame with factor data
        """
        try:
            from pandas_datareader import data as web
            
            if freq == "daily":
                if model == "3f":
                    df = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench")[0]
                elif model == "5f":
                    df = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench")[0]
            elif freq == "monthly":
                if model == "3f":
                    df = web.DataReader("F-F_Research_Data_Factors", "famafrench")[0]
                elif model == "5f":
                    df = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench")[0]
            else:
                raise ValueError(f"Unknown frequency: {freq}")
            
            # Convert to decimal (Fama-French returns in bps)
            return df / 100
        
        except Exception as e:
            print(f"Could not fetch Fama-French factors: {e}")
            return self._generate_synthetic_factors()

    def _generate_synthetic_factors(self, periods: int = 252) -> pd.DataFrame:
        """
        Generate synthetic factor data for testing.
        
        Args:
            periods: Number of periods to generate
            
        Returns:
            DataFrame with synthetic factors
        """
        dates = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq="D")
        
        return pd.DataFrame({
            "Mkt-RF": np.random.normal(0.0004, 0.01, periods),
            "SMB": np.random.normal(0.0001, 0.005, periods),
            "HML": np.random.normal(0.0001, 0.005, periods),
            "RF": np.full(periods, 0.00002),
        }, index=dates)

    # ========== REGRESSION & ANALYSIS ==========

    def fama_french_regression(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        include_constant: bool = True,
    ) -> Dict[str, object]:
        """
        Run Fama-French regression.
        
        Model: Return = alpha + beta_mkt*(Mkt-RF) + beta_smb*SMB + beta_hml*HML + epsilon
        
        Args:
            returns: Asset returns
            factors: Factor returns (Mkt-RF, SMB, HML, RF)
            include_constant: Include intercept (alpha)
            
        Returns:
            Regression results dictionary
        """
        # Align data
        valid_idx = returns.notna()
        returns_clean = returns[valid_idx].values
        
        # Prepare factors
        if include_constant:
            X = factors.loc[valid_idx, ["Mkt-RF", "SMB", "HML"]].values
            X = np.column_stack([np.ones(len(X)), X])
            factor_names = ["Alpha"] + ["Mkt-RF", "SMB", "HML"]
        else:
            X = factors.loc[valid_idx, ["Mkt-RF", "SMB", "HML"]].values
            factor_names = ["Mkt-RF", "SMB", "HML"]
        
        # OLS regression
        if len(returns_clean) < len(factor_names):
            raise ValueError("Insufficient data for regression")
        
        beta = np.linalg.lstsq(X, returns_clean, rcond=None)[0]
        residuals = returns_clean - X @ beta
        
        # Statistics
        n = len(returns_clean)
        k = X.shape[1]
        mse = np.sum(residuals**2) / (n - k)
        var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
        t_stats = beta / np.sqrt(var_beta)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
        
        # R-squared
        ss_tot = np.sum((returns_clean - returns_clean.mean())**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - ss_res / ss_tot
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
        
        return {
            "coefficients": dict(zip(factor_names, beta)),
            "t_stats": dict(zip(factor_names, t_stats)),
            "p_values": dict(zip(factor_names, p_values)),
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "residuals": residuals,
            "residual_std": np.std(residuals),
            "alpha": beta[0] if include_constant else 0,
            "alpha_annualized": beta[0] * 252 if include_constant else 0,
        }

    def fama_french_5f_regression(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
    ) -> Dict[str, object]:
        """
        Run Fama-French 5-factor regression.
        
        Model: Return = alpha + betas*Factors + epsilon
        (includes Mkt-RF, SMB, HML, RMW, CMA)
        
        Args:
            returns: Asset returns
            factors: Factor returns (must include all 5 factors)
            
        Returns:
            Regression results
        """
        valid_idx = returns.notna()
        returns_clean = returns[valid_idx].values
        
        required_factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        if not all(f in factors.columns for f in required_factors):
            raise ValueError(f"Missing factors. Need: {required_factors}")
        
        X = factors.loc[valid_idx, required_factors].values
        X = np.column_stack([np.ones(len(X)), X])
        factor_names = ["Alpha"] + required_factors
        
        beta = np.linalg.lstsq(X, returns_clean, rcond=None)[0]
        residuals = returns_clean - X @ beta
        
        n = len(returns_clean)
        k = X.shape[1]
        mse = np.sum(residuals**2) / (n - k)
        var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
        t_stats = beta / np.sqrt(var_beta)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
        
        ss_tot = np.sum((returns_clean - returns_clean.mean())**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - ss_res / ss_tot
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
        
        return {
            "coefficients": dict(zip(factor_names, beta)),
            "t_stats": dict(zip(factor_names, t_stats)),
            "p_values": dict(zip(factor_names, p_values)),
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "alpha_annualized": beta[0] * 252,
        }

    # ========== ORTHOGONALIZATION ==========

    def orthogonalize_factor(
        self,
        factor: pd.Series,
        control_factors: pd.DataFrame,
        method: str = "regression",
    ) -> pd.Series:
        """
        Orthogonalize factor against control factors.
        
        Removes common variation with known risk factors.
        
        Args:
            factor: Factor to orthogonalize
            control_factors: DataFrame of control factors
            method: 'regression' or 'gram_schmidt'
            
        Returns:
            Orthogonalized factor (residuals)
        """
        if method == "regression":
            # Run regression: factor = alpha + betas*controls + residuals
            valid_idx = factor.notna()
            
            X = control_factors.loc[valid_idx].values
            y = factor[valid_idx].values
            
            X = np.column_stack([np.ones(len(X)), X])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            
            return pd.Series(residuals, index=factor[valid_idx].index)
        
        elif method == "gram_schmidt":
            # Gram-Schmidt orthogonalization
            factor_clean = factor.dropna()
            controls_clean = control_factors.loc[factor_clean.index].dropna()
            
            # Start with factor
            orthogonal = factor_clean.values.copy().astype(float)
            
            # Subtract projections on each control factor
            for col in controls_clean.columns:
                control = controls_clean[col].values
                
                # Projection of factor onto control
                proj = np.dot(orthogonal, control) / np.dot(control, control) * control
                orthogonal = orthogonal - proj
            
            return pd.Series(orthogonal, index=factor_clean.index)
        
        else:
            raise ValueError(f"Unknown method: {method}")

    def gram_schmidt_orthogonalize(
        self,
        factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Orthogonalize multiple factors using Gram-Schmidt process.
        
        Args:
            factors: DataFrame of factors to orthogonalize
            
        Returns:
            DataFrame of orthogonalized factors
        """
        factors_clean = factors.dropna()
        orthogonal = factors_clean.values.copy().astype(float)
        
        for i in range(orthogonal.shape[1]):
            for j in range(i):
                # Remove component in direction of j-th orthogonal vector
                proj = (np.dot(orthogonal[:, i], orthogonal[:, j]) / 
                       np.dot(orthogonal[:, j], orthogonal[:, j]) * orthogonal[:, j])
                orthogonal[:, i] = orthogonal[:, i] - proj
            
            # Normalize
            orthogonal[:, i] = orthogonal[:, i] / np.linalg.norm(orthogonal[:, i])
        
        return pd.DataFrame(orthogonal, index=factors_clean.index, columns=factors.columns)

    # ========== CORRELATION & ANALYSIS ==========

    def factor_correlation_matrix(
        self,
        alpha_factor: pd.Series,
        known_factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate correlations between alpha and known factors.
        
        Args:
            alpha_factor: Generated alpha factor
            known_factors: Known risk factors (Mkt-RF, SMB, HML, etc.)
            
        Returns:
            Correlation matrix
        """
        combined = pd.concat([alpha_factor.rename("Alpha"), known_factors], axis=1)
        combined = combined.dropna()
        
        return combined.corr()

    def factor_exposure(
        self,
        alpha_factor: pd.Series,
        known_factors: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Measure exposure of alpha factor to known factors (loadings).
        
        Args:
            alpha_factor: Alpha factor values
            known_factors: Known factors
            
        Returns:
            Dictionary of factor loadings
        """
        results = self.fama_french_regression(alpha_factor, known_factors)
        
        exposures = {}
        for factor in ["Mkt-RF", "SMB", "HML"]:
            exposures[factor] = results["coefficients"].get(factor, 0)
        
        return exposures

    def idiosyncratic_variance(
        self,
        alpha_factor: pd.Series,
        known_factors: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Decompose alpha factor variance into factor and idiosyncratic components.
        
        Args:
            alpha_factor: Alpha factor
            known_factors: Known factors
            
        Returns:
            Dictionary with variance breakdown
        """
        results = self.fama_french_regression(alpha_factor, known_factors)
        
        total_var = np.var(alpha_factor.dropna())
        idiosync_var = np.var(results["residuals"])
        explained_var = total_var - idiosync_var
        
        return {
            "total_variance": total_var,
            "explained_variance": explained_var,
            "idiosyncratic_variance": idiosync_var,
            "r_squared": results["r_squared"],
            "idiosyncratic_pct": idiosync_var / total_var * 100,
        }

    # ========== REPORTING ==========

    def factor_analysis_report(
        self,
        alpha_factor: pd.Series,
        factor_returns: pd.DataFrame,
        alpha_name: str = "RL_Alpha",
    ) -> Dict[str, object]:
        """
        Generate comprehensive factor analysis report.
        
        Args:
            alpha_factor: Generated alpha factor
            factor_returns: Known factor returns
            alpha_name: Name for the alpha factor
            
        Returns:
            Complete analysis report
        """
        # Regression
        reg_results = self.fama_french_regression(alpha_factor, factor_returns)
        
        # Correlation
        corr_matrix = self.factor_correlation_matrix(alpha_factor, factor_returns)
        
        # Variance decomposition
        var_decomp = self.idiosyncratic_variance(alpha_factor, factor_returns)
        
        # Orthogonalized
        ortho_factor = self.orthogonalize_factor(
            alpha_factor,
            factor_returns[["Mkt-RF", "SMB", "HML"]]
        )
        
        report = {
            "factor_name": alpha_name,
            "regression": {
                "alpha": reg_results["alpha_annualized"],
                "alpha_t_stat": reg_results["t_stats"]["Alpha"],
                "mkt_beta": reg_results["coefficients"]["Mkt-RF"],
                "smb_beta": reg_results["coefficients"]["SMB"],
                "hml_beta": reg_results["coefficients"]["HML"],
                "r_squared": reg_results["r_squared"],
            },
            "correlations": corr_matrix.to_dict(),
            "variance_decomposition": var_decomp,
            "orthogonalized_factor": ortho_factor,
            "orthogonal_ic": corr_matrix.loc["Alpha", "Mkt-RF"],
        }
        
        return report

    def factor_comparison_table(
        self,
        alpha_factors: Dict[str, pd.Series],
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create comparison table for multiple alpha factors.
        
        Args:
            alpha_factors: Dictionary of factor_name → factor_series
            factor_returns: Known factor returns
            
        Returns:
            Comparison DataFrame
        """
        comparison_rows = []
        
        for name, factor in alpha_factors.items():
            reg = self.fama_french_regression(factor, factor_returns)
            var = self.idiosyncratic_variance(factor, factor_returns)
            
            comparison_rows.append({
                "Factor": name,
                "Alpha_Ann": reg["alpha_annualized"],
                "Alpha_tstat": reg["t_stats"]["Alpha"],
                "Mkt_Beta": reg["coefficients"]["Mkt-RF"],
                "SMB_Beta": reg["coefficients"]["SMB"],
                "HML_Beta": reg["coefficients"]["HML"],
                "R_Squared": reg["r_squared"],
                "Idio_Var_%": var["idiosyncratic_pct"],
            })
        
        return pd.DataFrame(comparison_rows)

    def print_factor_report(self, report: Dict[str, object]) -> None:
        """Pretty print factor analysis report."""
        print("\n" + "="*70)
        print(f"FACTOR ANALYSIS REPORT: {report['factor_name']}")
        print("="*70)
        
        print("\n[FAMA-FRENCH REGRESSION]")
        reg = report["regression"]
        print(f"  Alpha (ann.):      {reg['alpha']:8.4%}  (t = {reg['alpha_t_stat']:6.2f})")
        print(f"  Market Beta:       {reg['mkt_beta']:8.4f}")
        print(f"  SMB Beta:          {reg['smb_beta']:8.4f}")
        print(f"  HML Beta:          {reg['hml_beta']:8.4f}")
        print(f"  R-Squared:         {reg['r_squared']:8.4f}")
        
        print("\n[VARIANCE DECOMPOSITION]")
        var = report["variance_decomposition"]
        print(f"  Total Variance:    {var['total_variance']:8.6f}")
        print(f"  Explained:         {var['explained_variance']:8.6f} ({var['explained_variance']/var['total_variance']*100:.1f}%)")
        print(f"  Idiosyncratic:     {var['idiosyncratic_variance']:8.6f} ({var['idiosyncratic_pct']:.1f}%)")
        
        print("\n[FACTOR CORRELATIONS]")
        corr = report["correlations"]["Alpha"]
        for factor, corr_val in corr.items():
            if factor != "Alpha":
                print(f"  Corr with {factor:<10}: {corr_val:8.4f}")
        
        print("\n" + "="*70)
