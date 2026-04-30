"""
Mathematical operators for alpha factor construction.

Provides:
- Unary operations (ts_mean, ts_std, cs_rank, etc.)
- Binary arithmetic operations (+, -, *, /)
- Comparison operations
- Composition and evaluation framework
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class Operator:
    """
    Base class for all mathematical operators.
    
    Attributes:
        name: Operator identifier
        arity: Number of inputs (1 for unary, 2 for binary)
        complexity: Computational complexity score
        description: Human-readable description
    """

    def __init__(
        self,
        name: str,
        arity: int,
        complexity: int = 1,
        description: str = "",
    ):
        self.name = name
        self.arity = arity
        self.complexity = complexity
        self.description = description

    def __call__(self, *args, **kwargs):
        """Evaluate operator."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}(arity={self.arity})"


# ========== UNARY TIME-SERIES OPERATORS ==========

class TSMean(Operator):
    """Rolling time-series mean."""

    def __init__(self, window: int = 20):
        super().__init__(
            name="ts_mean",
            arity=1,
            complexity=2,
            description="Rolling mean over time window",
        )
        self.window = window

    def __call__(self, data: pd.Series) -> pd.Series:
        """Calculate rolling mean."""
        return data.rolling(window=self.window, min_periods=1).mean()


class TSStd(Operator):
    """Rolling time-series standard deviation."""

    def __init__(self, window: int = 20):
        super().__init__(
            name="ts_std",
            arity=1,
            complexity=2,
            description="Rolling standard deviation over time window",
        )
        self.window = window

    def __call__(self, data: pd.Series) -> pd.Series:
        """Calculate rolling std."""
        return data.rolling(window=self.window, min_periods=1).std()


class TSDelay(Operator):
    """Delay (lag) a time series."""

    def __init__(self, periods: int = 1):
        super().__init__(
            name="ts_delay",
            arity=1,
            complexity=1,
            description="Shift series by N periods",
        )
        self.periods = periods

    def __call__(self, data: pd.Series) -> pd.Series:
        """Lag the series."""
        return data.shift(periods=self.periods)


class TSDelta(Operator):
    """Period-to-period change (returns)."""

    def __init__(self, periods: int = 1):
        super().__init__(
            name="ts_delta",
            arity=1,
            complexity=1,
            description="Period-to-period change",
        )
        self.periods = periods

    def __call__(self, data: pd.Series) -> pd.Series:
        """Calculate returns."""
        return data.diff(periods=self.periods)


class TSLogReturn(Operator):
    """Log return calculation."""

    def __init__(self):
        super().__init__(
            name="ts_logret",
            arity=1,
            complexity=2,
            description="Logarithmic returns",
        )

    def __call__(self, data: pd.Series) -> pd.Series:
        """Calculate log returns."""
        return np.log(data / data.shift(1))


class TSMomentum(Operator):
    """Rate of change (momentum)."""

    def __init__(self, periods: int = 5):
        super().__init__(
            name="ts_momentum",
            arity=1,
            complexity=2,
            description="Momentum over N periods",
        )
        self.periods = periods

    def __call__(self, data: pd.Series) -> pd.Series:
        """Calculate momentum."""
        return (data - data.shift(self.periods)) / data.shift(self.periods)


# ========== UNARY CROSS-SECTIONAL OPERATORS ==========

class CSRank(Operator):
    """Cross-sectional ranking."""

    def __init__(self, pct: bool = True):
        super().__init__(
            name="cs_rank",
            arity=1,
            complexity=2,
            description="Cross-sectional percentile rank",
        )
        self.pct = pct

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Rank cross-sectionally."""
        return data.rank(axis=1, pct=self.pct)


class CSNormalize(Operator):
    """Cross-sectional normalization."""

    def __init__(self, method: str = "zscore"):
        super().__init__(
            name="cs_normalize",
            arity=1,
            complexity=2,
            description=f"Cross-sectional {method} normalization",
        )
        self.method = method

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize cross-sectionally."""
        if self.method == "zscore":
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            return (data - mean) / (std + 1e-8)
        elif self.method == "minmax":
            min_val = data.min(axis=1, keepdims=True)
            max_val = data.max(axis=1, keepdims=True)
            return (data - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization: {self.method}")


class CSScale(Operator):
    """Scale to fixed range."""

    def __init__(self, scale: float = 1.0):
        super().__init__(
            name="cs_scale",
            arity=1,
            complexity=1,
            description="Scale to fixed range",
        )
        self.scale = scale

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale data."""
        return data * self.scale


# ========== UNARY UTILITY OPERATORS ==========

class Abs(Operator):
    """Absolute value."""

    def __init__(self):
        super().__init__(
            name="abs",
            arity=1,
            complexity=1,
            description="Absolute value",
        )

    def __call__(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Take absolute value."""
        return data.abs()


class Sign(Operator):
    """Sign function (-1, 0, 1)."""

    def __init__(self):
        super().__init__(
            name="sign",
            arity=1,
            complexity=1,
            description="Sign of data (-1, 0, 1)",
        )

    def __call__(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Return sign."""
        return np.sign(data)


class Log(Operator):
    """Natural logarithm (safe)."""

    def __init__(self):
        super().__init__(
            name="log",
            arity=1,
            complexity=1,
            description="Natural logarithm",
        )

    def __call__(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Log with safety check."""
        return np.log(np.abs(data) + 1e-8)


class Sqrt(Operator):
    """Square root (safe)."""

    def __init__(self):
        super().__init__(
            name="sqrt",
            arity=1,
            complexity=1,
            description="Square root",
        )

    def __call__(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Sqrt with safety check."""
        return np.sqrt(np.abs(data))


# ========== BINARY ARITHMETIC OPERATORS ==========

class Add(Operator):
    """Addition."""

    def __init__(self):
        super().__init__(
            name="add",
            arity=2,
            complexity=1,
            description="Addition: A + B",
        )

    def __call__(
        self,
        data1: Union[pd.Series, pd.DataFrame],
        data2: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Add two operands."""
        return data1 + data2


class Subtract(Operator):
    """Subtraction."""

    def __init__(self):
        super().__init__(
            name="subtract",
            arity=2,
            complexity=1,
            description="Subtraction: A - B",
        )

    def __call__(
        self,
        data1: Union[pd.Series, pd.DataFrame],
        data2: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Subtract operands."""
        return data1 - data2


class Multiply(Operator):
    """Multiplication."""

    def __init__(self):
        super().__init__(
            name="multiply",
            arity=2,
            complexity=1,
            description="Multiplication: A * B",
        )

    def __call__(
        self,
        data1: Union[pd.Series, pd.DataFrame],
        data2: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Multiply operands."""
        return data1 * data2


class Divide(Operator):
    """Division (safe)."""

    def __init__(self):
        super().__init__(
            name="divide",
            arity=2,
            complexity=1,
            description="Division: A / B (safe)",
        )

    def __call__(
        self,
        data1: Union[pd.Series, pd.DataFrame],
        data2: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Divide with safety check."""
        return data1 / (data2 + 1e-8)


class Maximum(Operator):
    """Element-wise maximum."""

    def __init__(self):
        super().__init__(
            name="maximum",
            arity=2,
            complexity=1,
            description="Element-wise maximum",
        )

    def __call__(
        self,
        data1: Union[pd.Series, pd.DataFrame],
        data2: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Return element-wise max."""
        return np.maximum(data1, data2)


class Minimum(Operator):
    """Element-wise minimum."""

    def __init__(self):
        super().__init__(
            name="minimum",
            arity=2,
            complexity=1,
            description="Element-wise minimum",
        )

    def __call__(
        self,
        data1: Union[pd.Series, pd.DataFrame],
        data2: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Return element-wise min."""
        return np.minimum(data1, data2)


# ========== BINARY TIME-SERIES OPERATORS ==========

class TSCorr(Operator):
    """Rolling correlation between two series."""

    def __init__(self, window: int = 20):
        super().__init__(
            name="ts_corr",
            arity=2,
            complexity=3,
            description="Rolling correlation",
        )
        self.window = window

    def __call__(
        self,
        data1: pd.Series,
        data2: pd.Series,
    ) -> pd.Series:
        """Calculate rolling correlation."""
        return data1.rolling(window=self.window, min_periods=1).corr(data2)


class TSCovariance(Operator):
    """Rolling covariance between two series."""

    def __init__(self, window: int = 20):
        super().__init__(
            name="ts_cov",
            arity=2,
            complexity=3,
            description="Rolling covariance",
        )
        self.window = window

    def __call__(
        self,
        data1: pd.Series,
        data2: pd.Series,
    ) -> pd.Series:
        """Calculate rolling covariance."""
        result = []
        for i in range(len(data1)):
            if i < self.window:
                window_data1 = data1.iloc[:i+1]
                window_data2 = data2.iloc[:i+1]
            else:
                window_data1 = data1.iloc[i-self.window+1:i+1]
                window_data2 = data2.iloc[i-self.window+1:i+1]
            
            if len(window_data1) > 1:
                cov = np.cov(window_data1.values, window_data2.values)[0, 1]
                result.append(cov)
            else:
                result.append(np.nan)
        
        return pd.Series(result, index=data1.index)


class TSRatio(Operator):
    """Ratio of two time series (safe division)."""

    def __init__(self):
        super().__init__(
            name="ts_ratio",
            arity=2,
            complexity=2,
            description="Time-series ratio A/B",
        )

    def __call__(
        self,
        data1: pd.Series,
        data2: pd.Series,
    ) -> pd.Series:
        """Safe ratio calculation."""
        return data1 / (data2.abs() + 1e-8)


# ========== OPERATOR LIBRARY ==========

class OperatorLibrary:
    """
    Complete library of operators for RL-based formula generation.
    
    Provides:
    - Registry of all available operators
    - Operator statistics and complexity metrics
    - Convenient access by name or type
    """

    def __init__(self, window_sizes: Optional[List[int]] = None):
        """
        Initialize operator library.
        
        Args:
            window_sizes: Window sizes for rolling operators (default: [5, 10, 20])
        """
        self.window_sizes = window_sizes or [5, 10, 20]
        self.operators: Dict[str, Operator] = {}
        self._build_library()

    def _build_library(self):
        """Build complete operator library."""
        # Unary time-series
        for w in self.window_sizes:
            self.operators[f"ts_mean_{w}"] = TSMean(window=w)
            self.operators[f"ts_std_{w}"] = TSStd(window=w)
            self.operators[f"ts_momentum_{w}"] = TSMomentum(periods=w)
        
        for p in [1, 5, 10]:
            self.operators[f"ts_delay_{p}"] = TSDelay(periods=p)
            self.operators[f"ts_delta_{p}"] = TSDelta(periods=p)
        
        self.operators["ts_logret"] = TSLogReturn()
        
        # Unary cross-sectional
        self.operators["cs_rank"] = CSRank(pct=True)
        self.operators["cs_normalize_z"] = CSNormalize(method="zscore")
        self.operators["cs_normalize_mm"] = CSNormalize(method="minmax")
        self.operators["cs_scale"] = CSScale(scale=1.0)
        
        # Unary utilities
        self.operators["abs"] = Abs()
        self.operators["sign"] = Sign()
        self.operators["log"] = Log()
        self.operators["sqrt"] = Sqrt()
        
        # Binary arithmetic
        self.operators["add"] = Add()
        self.operators["subtract"] = Subtract()
        self.operators["multiply"] = Multiply()
        self.operators["divide"] = Divide()
        self.operators["maximum"] = Maximum()
        self.operators["minimum"] = Minimum()
        
        # Binary time-series
        for w in self.window_sizes:
            self.operators[f"ts_corr_{w}"] = TSCorr(window=w)
            self.operators[f"ts_cov_{w}"] = TSCovariance(window=w)
        
        self.operators["ts_ratio"] = TSRatio()

    def get(self, name: str) -> Optional[Operator]:
        """Get operator by name."""
        return self.operators.get(name)

    def get_by_arity(self, arity: int) -> Dict[str, Operator]:
        """Get all operators of a given arity (1=unary, 2=binary)."""
        return {name: op for name, op in self.operators.items() if op.arity == arity}

    def get_unary(self) -> Dict[str, Operator]:
        """Get all unary operators."""
        return self.get_by_arity(1)

    def get_binary(self) -> Dict[str, Operator]:
        """Get all binary operators."""
        return self.get_by_arity(2)

    def list_operators(self, arity: Optional[int] = None) -> List[str]:
        """List operator names."""
        if arity is None:
            return list(self.operators.keys())
        return list(self.get_by_arity(arity).keys())

    def operator_complexity(self, name: str) -> int:
        """Get complexity score of operator."""
        op = self.get(name)
        return op.complexity if op else float("inf")

    def get_complexity_stats(self) -> Dict[str, object]:
        """Get statistics on operator complexity."""
        complexities = [op.complexity for op in self.operators.values()]
        return {
            "total_operators": len(self.operators),
            "unary_operators": len(self.get_unary()),
            "binary_operators": len(self.get_binary()),
            "avg_complexity": np.mean(complexities),
            "max_complexity": np.max(complexities),
            "min_complexity": np.min(complexities),
        }

    def print_library(self):
        """Print library contents."""
        print("\n" + "="*70)
        print("OPERATOR LIBRARY")
        print("="*70)
        
        unary = self.get_unary()
        binary = self.get_binary()
        
        print(f"\nUnary Operators ({len(unary)}):")
        for name, op in sorted(unary.items()):
            print(f"  {name:<25} Complexity={op.complexity} | {op.description}")
        
        print(f"\nBinary Operators ({len(binary)}):")
        for name, op in sorted(binary.items()):
            print(f"  {name:<25} Complexity={op.complexity} | {op.description}")
        
        stats = self.get_complexity_stats()
        print(f"\nLibrary Statistics:")
        print(f"  Total Operators: {stats['total_operators']}")
        print(f"  Avg Complexity: {stats['avg_complexity']:.2f}")
        print("="*70 + "\n")
