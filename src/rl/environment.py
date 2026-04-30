"""
Reinforcement Learning Environment for Alpha Factor Discovery.

Defines the Gymnasium environment where RL agents construct mathematical expressions
(alpha factors) through sequential actions and learn to maximize risk-adjusted returns.

Environment Overview:
- Action Space: Discrete actions selecting operators and operands
- State Space: Tree properties + market entropy + performance metrics
- Reward: Information Coefficient - complexity penalty - turnover penalty
- Episode: Up to MAX_STEPS actions constructing a single expression
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List

from ..operator import OperatorLibrary, ExpressionBuilder, ExpressionTree
from ..evaluation import CrossSectionalBacktester, FactorAnalyzer
from .reward import RewardCalculator


class AlphaDiscoveryEnv(gym.Env):
    """
    Gymnasium environment for RL-based alpha factor discovery.
    
    The agent constructs expressions step-by-step by selecting:
    1. Input variables (price, volume, etc.)
    2. Unary operators (ts_mean, ts_momentum, cs_rank, etc.)
    3. Binary operators (add, multiply, divide, etc.)
    
    Rewards are based on:
    - Information Coefficient (IC) with forward returns
    - Complexity penalty (discourage overfitting)
    - Turnover penalty (prefer stable factors)
    
    Attributes:
        data (dict): Market data {asset_name: DataFrame with OHLV}
        returns (pd.DataFrame): Forward returns for reward calculation
        operator_library (OperatorLibrary): Available operators
        observation_space (spaces.Space): State representation
        action_space (spaces.Discrete): Available actions
    """
    
    metadata = {"render_modes": []}
    
    # Environment constraints
    MAX_STEPS = 20  # Maximum actions per episode
    MAX_HEIGHT = 10  # Maximum tree depth
    MAX_SIZE = 100  # Maximum nodes in tree
    MAX_COMPLEXITY = 50  # Maximum complexity score
    
    # Action types
    ACTION_ADD_INPUT = 0  # Start: add_input(name)
    ACTION_SELECT_UNARY = 1  # Then: apply_unary(op_name)
    ACTION_SELECT_BINARY = 2  # Then: apply_binary(op_name)
    ACTION_COMPLETE = 3  # End: evaluate and return reward
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        lookback_period: int = 30,
        forward_period: int = 5,
        window_sizes: List[int] = None,
        lambda_complexity: float = 0.1,
        lambda_turnover: float = 0.05,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the alpha discovery environment.
        
        Args:
            data: {asset_name: DataFrame with OHLCV columns}
            returns: Forward returns for evaluation (same index as data)
            lookback_period: Historical window for factor calculation
            forward_period: Forward window for return prediction
            window_sizes: Rolling window sizes for operators [5, 10, 20]
            lambda_complexity: Penalty weight for expression complexity
            lambda_turnover: Penalty weight for factor turnover
            render_mode: Visualization mode (not implemented)
        """
        self.data = data
        self.returns = returns
        self.lookback_period = lookback_period
        self.forward_period = forward_period
        self.window_sizes = window_sizes or [5, 10, 20]
        self.lambda_complexity = lambda_complexity
        self.lambda_turnover = lambda_turnover
        self.render_mode = render_mode
        
        # Initialize operator library
        self.operator_library = OperatorLibrary(window_sizes=self.window_sizes)
        self.reward_calculator = RewardCalculator(
            lambda_complexity=lambda_complexity,
            lambda_turnover=lambda_turnover,
        )
        
        # Extract available inputs and operators
        self.available_inputs = list(data.keys())
        self.unary_ops = list(self.operator_library.get_unary().keys())
        self.binary_ops = list(self.operator_library.get_binary().keys())
        
        # Action space: discrete actions for selecting from options
        # Action encoding:
        # 0-n: add_input(available_inputs[action])
        # n+1 to n+m: apply_unary(unary_ops[action])
        # n+m+1 to n+m+k: apply_binary(binary_ops[action])
        # n+m+k+1: complete expression
        
        self.num_input_actions = len(self.available_inputs)
        self.num_unary_actions = len(self.unary_ops)
        self.num_binary_actions = len(self.binary_ops)
        
        total_actions = (
            self.num_input_actions
            + self.num_unary_actions
            + self.num_binary_actions
            + 1  # complete action
        )
        
        self.action_space = spaces.Discrete(total_actions)
        
        # State space: normalized properties of current expression
        # [height, size, complexity, market_entropy, ic_estimate, valid_flag]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )
        
        # Initialize episode state
        self.reset()
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset environment for new episode.
        
        Returns:
            observation: Initial state
            info: Metadata dict
        """
        super().reset(seed=seed)
        
        # Start fresh expression builder
        self.builder = ExpressionBuilder(self.operator_library)
        self.current_node = None
        self.step_count = 0
        
        # Track expression building history
        self.expression_history = []
        self.reward_history = []
        
        # Initial observation
        observation = self._get_observation()
        info = {
            "step": 0,
            "expression": None,
            "valid": False,
            "market_entropy": self._compute_market_entropy(),
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one action in the environment.
        
        Args:
            action: Discrete action from action_space
        
        Returns:
            observation: New state after action
            reward: Reward for this step
            terminated: Episode finished (max steps or completion)
            truncated: Interaction limit (always False for this env)
            info: Metadata dict
        """
        self.step_count += 1
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        
        # Decode action
        if action < self.num_input_actions:
            # Add input
            input_name = self.available_inputs[action]
            self.current_node = self.builder.add_input(input_name)
            action_name = f"add_input({input_name})"
            
        elif action < self.num_input_actions + self.num_unary_actions:
            # Apply unary operator
            op_idx = action - self.num_input_actions
            op_name = self.unary_ops[op_idx]
            
            if self.current_node is None:
                reward = -1.0  # Invalid: no input yet
                info["error"] = "Unary operator requires input"
            else:
                self.current_node = self.builder.apply_unary(op_name, self.current_node)
                action_name = f"apply_unary({op_name})"
        
        elif action < self.num_input_actions + self.num_unary_actions + self.num_binary_actions:
            # Apply binary operator
            op_idx = (
                action - self.num_input_actions - self.num_unary_actions
            )
            op_name = self.binary_ops[op_idx]
            
            if self.current_node is None:
                reward = -1.0  # Invalid: no input yet
                info["error"] = "Binary operator requires operands"
            else:
                # For simplicity: create a duplicate node as second operand
                # In more sophisticated version: RL agent selects both operands
                self.current_node = self.builder.apply_binary(
                    op_name, self.current_node, self.current_node
                )
                action_name = f"apply_binary({op_name})"
        
        else:
            # Complete expression
            action_name = "complete"
            terminated = True
            
            if self.current_node is not None:
                try:
                    expr = self.builder.build(self.current_node)
                    
                    # Validate expression
                    if not expr.is_syntactically_valid():
                        reward = -1.0
                        info["error"] = "Invalid expression"
                    else:
                        # Evaluate expression and calculate reward
                        reward, metrics = self._evaluate_expression(expr)
                        info.update(metrics)
                except Exception as e:
                    reward = -1.0
                    info["error"] = str(e)
            else:
                reward = -1.0
                info["error"] = "No expression to complete"
        
        # Check constraints
        if self.current_node is not None:
            if self.current_node.height() > self.MAX_HEIGHT:
                reward = min(reward, -0.5)  # Penalty for exceeding height
                info["constraint_violated"] = "max_height"
            
            if self.current_node.size() > self.MAX_SIZE:
                reward = min(reward, -0.5)  # Penalty for exceeding size
                info["constraint_violated"] = "max_size"
        
        # Check step limit
        if self.step_count >= self.MAX_STEPS:
            terminated = True
        
        # Record history
        self.expression_history.append(action_name)
        self.reward_history.append(reward)
        
        # Get next observation
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, info
    
    def _evaluate_expression(self, expr: ExpressionTree) -> Tuple[float, dict]:
        """
        Evaluate an expression and calculate reward.
        
        Args:
            expr: Expression tree to evaluate
        
        Returns:
            reward: Combined reward score
            metrics: Dictionary of individual metrics
        """
        try:
            # Get factor values from expression
            factor = expr.evaluate(self.data)
            
            if isinstance(factor, dict):
                # Multi-input factor - take first available
                factor = next(iter(factor.values()))
            
            # Calculate metrics
            metrics = {}
            
            # Information Coefficient (IC)
            aligned_returns = self.returns.loc[factor.index]
            ic = self._calculate_ic(factor, aligned_returns)
            metrics["ic"] = float(ic)
            
            # Complexity penalty
            complexity = expr.root.complexity()
            metrics["complexity"] = int(complexity)
            
            # Turnover (factor stability)
            turnover = self._calculate_turnover(factor)
            metrics["turnover"] = float(turnover)
            
            # Reward function: IC - lambda_complexity * complexity - lambda_turnover * turnover
            reward = (
                ic
                - self.lambda_complexity * (complexity / self.MAX_COMPLEXITY)
                - self.lambda_turnover * turnover
            )
            
            metrics["reward"] = float(reward)
            metrics["valid"] = True
            
            return reward, metrics
        
        except Exception as e:
            return -1.0, {"error": str(e), "valid": False}
    
    def _calculate_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        """Calculate Information Coefficient (Spearman rank correlation)."""
        from scipy.stats import spearmanr
        
        # Remove NaN values
        valid_idx = factor.notna() & returns.notna()
        if valid_idx.sum() < 10:
            return 0.0
        
        factor_clean = factor[valid_idx]
        returns_clean = returns[valid_idx]
        
        corr, _ = spearmanr(factor_clean.rank(), returns_clean.rank())
        return float(corr) if not np.isnan(corr) else 0.0
    
    def _calculate_turnover(self, factor: pd.Series, quantiles: int = 5) -> float:
        """
        Calculate portfolio turnover (average rebalancing costs).
        
        Turnover is the fraction of holdings that change between periods.
        """
        # Assign to quantile buckets
        quantile_groups = pd.qcut(factor, q=quantiles, labels=False, duplicates="drop")
        
        # Calculate turnover as average shift in group membership
        shifts = quantile_groups.diff().abs()
        turnover = shifts.sum() / len(shifts) if len(shifts) > 0 else 0.0
        
        return float(turnover)
    
    def _compute_market_entropy(self) -> float:
        """
        Compute market entropy as a feature of environment state.
        
        Higher entropy = more random market behavior = harder factor discovery.
        """
        try:
            # Use returns distribution entropy
            all_returns = pd.concat(
                [self.data[name]["Close"].pct_change() for name in self.data.keys()],
                axis=1
            )
            
            # Compute entropy as normalized variance of returns
            entropy = all_returns.std().std()
            return float(np.clip(entropy, 0, 1))
        except:
            return 0.5
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state) as normalized vector.
        
        State includes:
        - Tree height (normalized to 0-1)
        - Tree size (normalized to 0-1)
        - Complexity (normalized to 0-1)
        - Market entropy (0-1)
        - Step count (normalized to 0-1)
        - Valid flag (0 or 1)
        """
        if self.current_node is None:
            height = 0.0
            size = 0.0
            complexity = 0.0
            valid = 0.0
        else:
            height = float(self.current_node.height()) / self.MAX_HEIGHT
            size = float(self.current_node.size()) / self.MAX_SIZE
            complexity = float(self.current_node.complexity()) / self.MAX_COMPLEXITY
            valid = 1.0 if self.current_node.is_valid() else 0.0
        
        entropy = self._compute_market_entropy()
        step_norm = float(self.step_count) / self.MAX_STEPS
        
        observation = np.array(
            [height, size, complexity, entropy, step_norm, valid],
            dtype=np.float32,
        )
        
        return observation
    
    def render(self):
        """Rendering not implemented."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_expression_summary(self) -> dict:
        """Get summary of current expression."""
        if self.current_node is None:
            return {"summary": None, "valid": False}
        
        try:
            expr = self.builder.build(self.current_node)
            return {
                "formula": expr.to_string(),
                "height": expr.root.height(),
                "size": expr.root.size(),
                "complexity": expr.root.complexity(),
                "inputs": list(expr.root.get_all_inputs()),
                "valid": expr.is_syntactically_valid(),
            }
        except Exception as e:
            return {"error": str(e), "valid": False}


class PortfolioEnv(gym.Env):
    """
    Alternative environment for portfolio-level optimization.
    
    The agent constructs multiple factors and combines them into a portfolio.
    (Advanced variant - for future enhancement)
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame], returns: pd.DataFrame):
        """Initialize portfolio environment."""
        raise NotImplementedError("PortfolioEnv is planned for future release")
