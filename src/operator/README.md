# Operator Module Documentation

## Overview

The `src/operator` module provides the mathematical operator library and expression tree framework for RL-based formula generation. It enables the RL agent to construct complex mathematical expressions (DAGs) representing alpha factors.

## Architecture

```
src/operator/
├── __init__.py              # Module exports
├── math_ops.py              # Mathematical operators and library
├── expression_tree.py       # Expression tree, DAG, and code generation
└── example.py               # Example usage and RL integration patterns
```

## Components

### 1. Mathematical Operators (`math_ops.py`)

Complete library of mathematical building blocks for formula construction.

#### Operator Types:

**Unary Time-Series Operators:**
- `ts_mean_{window}`: Rolling mean over window (5, 10, 20 periods)
- `ts_std_{window}`: Rolling standard deviation
- `ts_momentum_{window}`: Rate of change momentum
- `ts_delay_{periods}`: Shift/lag by N periods (1, 5, 10)
- `ts_delta_{periods}`: Period-to-period change (returns)
- `ts_logret`: Logarithmic returns

**Unary Cross-Sectional Operators:**
- `cs_rank`: Cross-sectional percentile rank (0-1)
- `cs_normalize_z`: Z-score normalization
- `cs_normalize_mm`: Min-max normalization
- `cs_scale`: Scale to fixed range

**Unary Utility Operators:**
- `abs`: Absolute value
- `sign`: Sign function (-1, 0, 1)
- `log`: Natural logarithm (safe)
- `sqrt`: Square root (safe)

**Binary Arithmetic Operators:**
- `add`: Addition (A + B)
- `subtract`: Subtraction (A - B)
- `multiply`: Multiplication (A * B)
- `divide`: Division (A / B, safe)
- `maximum`: Element-wise maximum
- `minimum`: Element-wise minimum

**Binary Time-Series Operators:**
- `ts_corr_{window}`: Rolling correlation (5, 10, 20 periods)
- `ts_cov_{window}`: Rolling covariance
- `ts_ratio`: Safe division of time series

#### Usage:

```python
from src.operator import OperatorLibrary, TSMean

# Direct operator usage
ts_mean = TSMean(window=20)
result = ts_mean(price_series)

# Operator library
library = OperatorLibrary(window_sizes=[5, 10, 20])

# Get operator by name
op = library.get("ts_mean_20")

# List all operators
unary = library.get_unary()
binary = library.get_binary()

# Get operator statistics
stats = library.get_complexity_stats()
# Returns: {total_operators, unary_operators, binary_operators, avg_complexity, max_complexity}
```

---

### 2. Expression Trees (`expression_tree.py`)

Expression tree framework for building and evaluating complex formulas.

#### Key Concepts:

**ExprNode**: Single node in expression tree
- Leaf node: Raw data input (e.g., "price", "volume")
- Operator node: Applies operator to child nodes

**ExpressionTree**: Complete formula representation
- Validates syntactic correctness
- Checks for cycles (valid DAG)
- Evaluates on market data
- Generates Python code

#### Building Expressions Programmatically:

```python
from src.operator import ExpressionBuilder, OperatorLibrary

library = OperatorLibrary()
builder = ExpressionBuilder(library)

# Build: ts_mean_20(price)
price_input = builder.add_input("price")
ts_mean_node = builder.apply_unary("ts_mean_20", price_input)
expr = builder.build(ts_mean_node)

# Evaluate
result = expr.evaluate({"price": price_series})
```

#### Complex Expressions:

```python
# Build: (price - ts_mean_20(price)) / ts_std_20(price)
# (Zscore normalization)
price = builder.add_input("price")
mean = builder.apply_unary("ts_mean_20", price)
std = builder.apply_unary("ts_std_20", price)
deviation = builder.apply_binary("subtract", price, mean)
normalized = builder.apply_binary("divide", deviation, std)

expr = builder.build(normalized)
```

#### Evaluation:

```python
# Single-input evaluation
result = expr.evaluate({"price": price_series})

# Multi-input evaluation
result = expr.evaluate({
    "price": price_series,
    "volume": volume_series,
})

# With caching for complex expressions
result = expr.evaluate(data_dict, cache_results=True)
```

#### Tree Inspection:

```python
# Get tree properties
expr.root.height()              # Tree depth
expr.root.size()                # Number of nodes
expr.root.complexity()          # Total complexity score
expr.root.get_all_inputs()      # Set of input names

# Validate
expr.is_syntactically_valid()   # Check all nodes valid
expr.is_valid_dag()             # Check no cycles
expr.get_validation_status()    # Full status dict

# Convert to string
formula_str = expr.to_string()  # Human-readable formula
# Example: "(((price / ts_mean_20(price)) * ts_std_20(price)))"
```

#### Code Generation:

```python
# Generate Python function from expression
code = expr.to_python_code(func_name="alpha_factor")

# Returns executable Python code:
# def alpha_factor(price):
#     _expr_1 = ts_mean_20(price)
#     _expr_2 = divide(price, _expr_1)
#     return _expr_2
```

#### Serialization:

```python
# Convert to JSON
json_str = expr.to_json()

# Restore from JSON
expr_restored = ExpressionTree.from_json(json_str, library)

# Get expression hash (unique formula signature)
formula_hash = expr.root.hash()  # 8-character hash
```

#### Summary:

```python
summary = expr.summary()
# Returns:
# {
#   "formula": "(price + ts_mean_20(price))",
#   "height": 2,
#   "size": 3,
#   "complexity": 3,
#   "inputs": ["price"],
#   "hash": "a1b2c3d4",
#   "validation": {...}
# }

expr.print_summary()  # Pretty print summary
```

---

## Integration with RL Environment

### Typical RL Workflow:

```python
from src.operator import OperatorLibrary, ExpressionBuilder

# Initialize library with configurable windows
library = OperatorLibrary(window_sizes=[5, 10, 20])

# RL agent control loop
while training:
    # Get current expression
    builder = ExpressionBuilder(library)
    current_node = None
    
    while not episode_done:
        # RL agent chooses action
        action = agent.select_action(state)
        
        # Apply action to expression
        if action_type == "add_input":
            current_node = builder.add_input(action_value)
        elif action_type == "apply_unary":
            current_node = builder.apply_unary(action_value, current_node)
        elif action_type == "apply_binary":
            current_node = builder.apply_binary(action_value, left_node, right_node)
        
        # Build and evaluate expression
        expr = builder.build(current_node)
        
        # Calculate metrics
        factor = expr.evaluate(data)
        ic = compute_information_coefficient(factor, returns)
        complexity_cost = expr.root.complexity()
        
        # Reward function
        reward = ic - lambda_complexity * complexity_cost - lambda_turnover * turnover
        
        # Update policy
        agent.update(state, action, reward)
```

### Action Space for RL Agent:

```python
library = OperatorLibrary()

# Discrete action space
unary_operators = library.list_operators(arity=1)  # ~40 operators
binary_operators = library.list_operators(arity=2)  # ~10 operators
input_names = ["price", "volume", ...]              # ~5 inputs

# Total action space = unary + binary * 2 + inputs
```

### State Space Representation:

```python
# Partial expression state:
# - Current tree height (0-10)
# - Current tree size (0-100 nodes)
# - Current complexity (0-50)
# - Available inputs (binary mask)
# - Recent performance metrics (IC, Sharpe)
# - Market entropy (volatility dispersion)
```

### Constraints for RL Agent:

```python
# Maximum tree size
MAX_HEIGHT = 10
MAX_SIZE = 100
MAX_COMPLEXITY = 50

# Validity checks
assert expr.is_syntactically_valid()
assert expr.is_valid_dag()
assert expr.root.height() <= MAX_HEIGHT
assert expr.root.size() <= MAX_SIZE
assert expr.root.complexity() <= MAX_COMPLEXITY
```

---

## Operator Complexity Reference

Used for regularization in RL reward function:

| Operator Type | Complexity | Notes |
|---------------|-----------|-------|
| Input | 0 | No computation |
| Arithmetic | 1 | +, -, *, / |
| Delay | 1 | ts_delay |
| Compare | 1 | min, max |
| Unary Util | 1 | abs, sign |
| Time-Series (simple) | 2 | ts_mean, ts_std, ts_momentum |
| Normalization | 2 | cs_rank, cs_normalize |
| Utility (complex) | 2 | log, sqrt |
| Time-Series (advanced) | 3 | ts_corr, ts_covariance |

**Total Expression Complexity**: Sum of operator complexities in tree

---

## Safety Features

All operators have built-in safety checks:

```python
# Division by zero protection
divide(a, b)  # Returns a / (b + 1e-8)

# Log of negative/zero protection
log(data)  # Returns log(abs(data) + 1e-8)

# Square root safety
sqrt(data)  # Takes sqrt(abs(data))

# Missing data handling
ts_mean(series)  # min_periods=1, handles NaN
```

---

## Performance Characteristics

### Evaluation Speed:
- Simple expression (3-5 nodes): ~1-2 ms
- Complex expression (50+ nodes): ~10-20 ms
- Cross-sectional ops on 1000 assets: ~100-200 ms

### Memory Usage:
- Expression tree: ~100 bytes per node
- Large expression (100 nodes): ~10 KB
- Evaluation cache: depends on data size

---

## Examples

See `src/operator/example.py` for:
1. Operator library exploration
2. Simple expression building
3. Arithmetic expressions
4. Multi-input expressions
5. Cross-sectional expressions
6. Complex nested expressions
7. Python code generation
8. JSON serialization
9. RL integration pattern
10. Operator statistics

Run examples:
```bash
cd src/operator
python example.py
```

---

## Building Custom Operators

To extend the operator library:

```python
from src.operator import Operator
import pandas as pd

class CustomMomentum(Operator):
    def __init__(self, periods: int = 5):
        super().__init__(
            name="custom_momentum",
            arity=1,
            complexity=2,
            description="Custom momentum calculation",
        )
        self.periods = periods
    
    def __call__(self, data: pd.Series) -> pd.Series:
        return (data / data.shift(self.periods) - 1) * 100
```

---

## Requirements

- `pandas`: Time-series and DataFrame operations
- `numpy`: Numerical computations

---

## API Reference

### ExpressionTree

| Method | Description |
|--------|-------------|
| `evaluate(data)` | Compute factor from data |
| `is_syntactically_valid()` | Check all nodes valid |
| `is_valid_dag()` | Check no cycles |
| `get_validation_status()` | Full validation report |
| `to_string()` | Human-readable formula |
| `to_json()` | JSON serialization |
| `to_python_code(func_name)` | Generate Python code |
| `summary()` | Statistics dictionary |
| `print_summary()` | Pretty print summary |

### ExprNode

| Method | Description |
|--------|-------------|
| `is_leaf()` | Check if leaf node |
| `is_valid()` | Check syntactic validity |
| `height()` | Tree depth |
| `size()` | Number of nodes |
| `complexity()` | Total complexity |
| `get_all_inputs()` | Input names used |
| `to_string()` | Formula string |
| `hash()` | Expression hash |

### ExpressionBuilder

| Method | Description |
|--------|-------------|
| `add_input(name)` | Add data input node |
| `apply_unary(op_name, child)` | Apply unary operator |
| `apply_binary(op_name, left, right)` | Apply binary operator |
| `build(output_node)` | Create ExpressionTree |
| `clear()` | Reset builder |

---

## Future Enhancements

- [ ] Support for conditional operators (if-then-else)
- [ ] Symbolic optimization (simplify expressions)
- [ ] Automatic differentiation for gradients
- [ ] GPU acceleration for large expressions
- [ ] Expression pruning and simplification
- [ ] Multi-output expressions (for portfolio construction)
