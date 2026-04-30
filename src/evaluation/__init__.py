"""
Evaluation module for backtest analysis and factor orthogonalization.

Provides utilities for:
- Cross-sectional backtesting with turnover analysis
- Performance metrics (Sharpe, IC, Returns)
- Fama-French factor regression and orthogonalization
- Risk attribution and factor correlation
"""

from .backtester import CrossSectionalBacktester
from .factor_analysis import FactorAnalyzer

__all__ = ["CrossSectionalBacktester", "FactorAnalyzer"]
