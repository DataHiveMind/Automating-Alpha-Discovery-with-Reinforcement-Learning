"""
Operator module for expression tree construction and evaluation.

Provides:
- Mathematical operators (unary, binary, time-series, cross-sectional)
- Expression tree and DAG generation
- Formula parsing and code generation
- Operator library with complexity tracking
"""

from .math_ops import (
    Operator,
    OperatorLibrary,
    # Unary time-series
    TSMean,
    TSStd,
    TSDelay,
    TSDelta,
    TSLogReturn,
    TSMomentum,
    # Unary cross-sectional
    CSRank,
    CSNormalize,
    CSScale,
    # Unary utilities
    Abs,
    Sign,
    Log,
    Sqrt,
    # Binary arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    # Binary time-series
    TSCorr,
    TSCovariance,
    TSRatio,
)

from .expression_tree import (
    ExprNode,
    ExpressionTree,
    ExpressionBuilder,
)

__all__ = [
    # Core classes
    "Operator",
    "OperatorLibrary",
    "ExprNode",
    "ExpressionTree",
    "ExpressionBuilder",
    # Operators
    "TSMean",
    "TSStd",
    "TSDelay",
    "TSDelta",
    "TSLogReturn",
    "TSMomentum",
    "CSRank",
    "CSNormalize",
    "CSScale",
    "Abs",
    "Sign",
    "Log",
    "Sqrt",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Maximum",
    "Minimum",
    "TSCorr",
    "TSCovariance",
    "TSRatio",
]
