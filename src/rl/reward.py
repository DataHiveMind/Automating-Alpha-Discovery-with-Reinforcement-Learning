"""
Reward calculation for alpha factor discovery RL environment.

Defines multi-component reward functions that balance:
- Predictive power (Information Coefficient)
- Computational complexity (parsimony)
- Turnover (transaction costs)
- Risk metrics (Sharpe ratio, max drawdown)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from scipy import stats


class RewardCalculator:
    """
    Multi-component reward function for RL agent.
    
    Reward = f(IC, Complexity, Turnover, Sharpe, MaxDD)
    
    Weights balance exploration vs exploitation and discovery of robust factors.
    
    Attributes:
        lambda_complexity: Penalty for expression complexity
        lambda_turnover: Penalty for factor turnover
        lambda_sharpe: Weight for Sharpe ratio component
        lambda_maxdd: Penalty for maximum drawdown
    """
    
    def __init__(
        self,
        lambda_complexity: float = 0.1,
        lambda_turnover: float = 0.05,
        lambda_sharpe: float = 0.1,
        lambda_maxdd: float = 0.02,
        ic_target: float = 0.05,
        sharpe_target: float = 1.0,
    ):
        """
        Initialize reward calculator.
        
        Args:
            lambda_complexity: Penalty weight for complexity (0-1)
            lambda_turnover: Penalty weight for turnover (0-1)
            lambda_sharpe: Weight for Sharpe ratio (0-1)
            lambda_maxdd: Penalty weight for max drawdown (0-1)
            ic_target: Target IC for scaling (benchmark)
            sharpe_target: Target Sharpe for scaling (benchmark)
        """
        self.lambda_complexity = lambda_complexity
        self.lambda_turnover = lambda_turnover
        self.lambda_sharpe = lambda_sharpe
        self.lambda_maxdd = lambda_maxdd
        self.ic_target = ic_target
        self.sharpe_target = sharpe_target
    
    def calculate_reward(
        self,
        ic: float,
        complexity: float,
        turnover: float,
        sharpe: float = None,
        max_dd: float = None,
        max_complexity: float = 50.0,
    ) -> float:
        """
        Calculate multi-component reward.
        
        Reward = IC - λ_c * (complexity/max) - λ_t * turnover 
                 + λ_s * Sharpe - λ_dd * max_drawdown
        
        Args:
            ic: Information Coefficient (rank correlation with returns)
            complexity: Expression tree complexity score (0-50)
            turnover: Portfolio turnover (0-1, lower is better)
            sharpe: Sharpe ratio (optional)
            max_dd: Maximum drawdown (optional, negative value)
            max_complexity: Maximum complexity for normalization
        
        Returns:
            reward: Total reward score
        """
        # IC component (main objective - range: -1 to +1)
        ic_reward = ic
        
        # Complexity penalty (discourage overfitting)
        complexity_penalty = self.lambda_complexity * (complexity / max_complexity)
        
        # Turnover penalty (discourage excessive rebalancing)
        turnover_penalty = self.lambda_turnover * turnover
        
        # Base reward
        reward = ic_reward - complexity_penalty - turnover_penalty
        
        # Optional: Sharpe ratio bonus
        if sharpe is not None:
            sharpe_component = self.lambda_sharpe * (sharpe / self.sharpe_target)
            reward += np.clip(sharpe_component, -0.5, 0.5)  # Cap bonus
        
        # Optional: Drawdown penalty
        if max_dd is not None:
            # max_dd is negative (e.g., -0.15 for -15%)
            dd_penalty = self.lambda_maxdd * abs(max_dd)
            reward -= dd_penalty
        
        return float(reward)
    
    def calculate_ic(
        self,
        factor: pd.Series,
        returns: pd.Series,
        method: str = "spearman",
    ) -> float:
        """
        Calculate Information Coefficient.
        
        IC measures predictive power as correlation between factor and future returns.
        
        Args:
            factor: Factor values (cross-sectional or time-series)
            returns: Forward returns aligned with factor dates
            method: "spearman" (default) or "pearson"
        
        Returns:
            ic: Information Coefficient (-1 to +1)
        """
        # Remove NaN values
        valid_idx = factor.notna() & returns.notna()
        if valid_idx.sum() < 10:
            return 0.0
        
        factor_clean = factor[valid_idx]
        returns_clean = returns[valid_idx]
        
        if method == "spearman":
            # Rank-based correlation (more robust to outliers)
            corr, _ = stats.spearmanr(factor_clean.rank(), returns_clean.rank())
        else:
            # Pearson correlation
            corr, _ = stats.pearsonr(factor_clean, returns_clean)
        
        return float(corr) if not np.isnan(corr) else 0.0
    
    def calculate_turnover(
        self,
        factor: pd.Series,
        quantiles: int = 5,
    ) -> float:
        """
        Calculate portfolio turnover.
        
        Turnover measures the fraction of holdings that change between rebalances.
        Lower turnover = lower transaction costs.
        
        Args:
            factor: Factor values (ranked into quintiles)
            quantiles: Number of buckets for ranking
        
        Returns:
            turnover: Average turnover per period (0-1)
        """
        try:
            # Assign to quantile buckets
            quantile_groups = pd.qcut(
                factor, 
                q=quantiles, 
                labels=False, 
                duplicates="drop"
            )
            
            # Calculate shifts in group membership
            group_shifts = quantile_groups.diff().abs()
            
            # Turnover = fraction of holdings that change
            turnover = group_shifts.sum() / len(group_shifts) if len(group_shifts) > 0 else 0.0
            
            return float(np.clip(turnover, 0.0, 1.0))
        except:
            return 0.5  # Default if calculation fails
    
    def calculate_sharpe_ratio(
        self,
        factor: pd.Series,
        returns: pd.Series,
        quantiles: int = 5,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Sharpe ratio of long-short portfolio.
        
        Constructs a factor-based portfolio:
        - Long highest quantile (top performers)
        - Short lowest quantile (worst performers)
        - Equal-weight within quantiles
        
        Args:
            factor: Factor values
            returns: Asset returns (same index as factor)
            quantiles: Number of buckets
            periods_per_year: Trading periods per year (252 for daily)
        
        Returns:
            sharpe: Annualized Sharpe ratio
        """
        try:
            # Assign to quantiles
            quantile_groups = pd.qcut(
                factor,
                q=quantiles,
                labels=False,
                duplicates="drop"
            )
            
            # Portfolio returns
            n_quantiles = quantile_groups.max() + 1
            portfolio_returns = pd.Series(index=returns.index, dtype=float)
            
            for t in range(len(returns)):
                # Long top quantile, short bottom
                long_mask = quantile_groups.iloc[t] == (n_quantiles - 1)
                short_mask = quantile_groups.iloc[t] == 0
                
                long_return = returns.iloc[t][long_mask].mean()
                short_return = returns.iloc[t][short_mask].mean()
                
                portfolio_returns.iloc[t] = long_return - short_return
            
            # Calculate Sharpe ratio
            avg_return = portfolio_returns.mean() * periods_per_year
            std_return = portfolio_returns.std() * np.sqrt(periods_per_year)
            
            sharpe = avg_return / std_return if std_return > 0 else 0.0
            
            return float(sharpe)
        except:
            return 0.0
    
    def calculate_max_drawdown(
        self,
        factor: pd.Series,
        returns: pd.Series,
        quantiles: int = 5,
    ) -> float:
        """
        Calculate maximum drawdown of long-short portfolio.
        
        Maximum drawdown is the largest peak-to-trough decline.
        More negative value = worse (larger drawdown).
        
        Args:
            factor: Factor values
            returns: Asset returns
            quantiles: Number of buckets
        
        Returns:
            max_drawdown: Maximum drawdown (negative value, e.g., -0.15 for -15%)
        """
        try:
            # Get portfolio returns (same as Sharpe calculation)
            quantile_groups = pd.qcut(
                factor,
                q=quantiles,
                labels=False,
                duplicates="drop"
            )
            
            n_quantiles = quantile_groups.max() + 1
            portfolio_returns = pd.Series(index=returns.index, dtype=float)
            
            for t in range(len(returns)):
                long_mask = quantile_groups.iloc[t] == (n_quantiles - 1)
                short_mask = quantile_groups.iloc[t] == 0
                
                long_return = returns.iloc[t][long_mask].mean()
                short_return = returns.iloc[t][short_mask].mean()
                
                portfolio_returns.iloc[t] = long_return - short_return
            
            # Calculate cumulative returns and drawdown
            cum_returns = (1 + portfolio_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            
            max_dd = drawdown.min()
            
            return float(max_dd)
        except:
            return 0.0
    
    def calculate_factor_analysis(
        self,
        factor: pd.Series,
        returns: pd.Series,
        max_complexity: float = 50.0,
    ) -> Dict[str, float]:
        """
        Comprehensive factor analysis with all metrics.
        
        Args:
            factor: Factor values
            returns: Asset returns
            max_complexity: For normalization
        
        Returns:
            Dictionary with all metrics
        """
        ic = self.calculate_ic(factor, returns)
        turnover = self.calculate_turnover(factor)
        sharpe = self.calculate_sharpe_ratio(factor, returns)
        max_dd = self.calculate_max_drawdown(factor, returns)
        
        return {
            "ic": ic,
            "turnover": turnover,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "ic_is_significant": self._is_ic_significant(ic),
            "sharpe_is_positive": sharpe > 0.0,
            "factor_quality": self._factor_quality_score(ic, sharpe, max_dd),
        }
    
    def _is_ic_significant(self, ic: float, threshold: float = 0.03) -> bool:
        """Check if IC is statistically significant."""
        return abs(ic) > threshold
    
    def _factor_quality_score(
        self,
        ic: float,
        sharpe: float,
        max_dd: float,
    ) -> float:
        """
        Compute composite factor quality score.
        
        Combines IC, Sharpe, and drawdown into 0-100 score.
        """
        # Normalize components to 0-1
        ic_score = min(abs(ic) / 0.1, 1.0)  # Target IC = 0.1
        sharpe_score = min(sharpe / 1.0, 1.0)  # Target Sharpe = 1.0
        dd_score = min(1.0 + max_dd, 1.0)  # Penalize large drawdowns
        
        # Composite score (0-100)
        quality = (0.4 * ic_score + 0.4 * sharpe_score + 0.2 * dd_score) * 100
        
        return float(quality)


class ShapedRewardCalculator(RewardCalculator):
    """
    Reward shaping for curriculum learning.
    
    Provides intermediate rewards to accelerate learning:
    - Bonus for valid expressions
    - Bonus for reaching milestones (depth, IC threshold)
    - Progressive curriculum (easier targets first, then harder)
    """
    
    def __init__(
        self,
        *args,
        use_potential_shaping: bool = True,
        milestone_bonuses: bool = True,
        **kwargs
    ):
        """Initialize shaped reward calculator."""
        super().__init__(*args, **kwargs)
        self.use_potential_shaping = use_potential_shaping
        self.milestone_bonuses = milestone_bonuses
        self.previous_state = None
    
    def shaped_reward(
        self,
        ic: float,
        complexity: float,
        turnover: float,
        current_state: Dict[str, float],
        previous_state: Dict[str, float] = None,
        max_complexity: float = 50.0,
    ) -> float:
        """
        Calculate shaped reward with intermediate bonuses.
        
        Args:
            ic: Information Coefficient
            complexity: Expression complexity
            turnover: Portfolio turnover
            current_state: Current state metrics
            previous_state: Previous state for potential shaping
            max_complexity: Maximum complexity
        
        Returns:
            reward: Base + shaped components
        """
        # Base reward
        base_reward = self.calculate_reward(
            ic, complexity, turnover, max_complexity=max_complexity
        )
        
        # Validity bonus
        validity_bonus = 0.1 if current_state.get("valid", False) else 0.0
        
        # Milestone bonuses
        milestone_bonus = 0.0
        if self.milestone_bonuses:
            # Bonus for reaching complexity targets
            if complexity > 5 and (previous_state is None or previous_state.get("complexity", 0) <= 5):
                milestone_bonus += 0.05
            
            # Bonus for positive IC
            if ic > 0 and (previous_state is None or previous_state.get("ic", 0) <= 0):
                milestone_bonus += 0.1
        
        # Potential shaping
        potential_bonus = 0.0
        if self.use_potential_shaping and previous_state is not None:
            # Reward improvement in state features
            potential_bonus = 0.05 * (
                (current_state.get("ic", 0) - previous_state.get("ic", 0))
            )
        
        shaped_reward = base_reward + validity_bonus + milestone_bonus + potential_bonus
        
        return float(shaped_reward)
