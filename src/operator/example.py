"""
Example and integration script for the operator module.

Demonstrates:
- Operator library and usage
- Expression tree construction
- Formula evaluation and code generation
- RL environment integration
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from math_ops import OperatorLibrary
from expression_tree import ExprNode, ExpressionTree, ExpressionBuilder


def example_operator_library():
    """Example: Explore the operator library."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Operator Library")
    print("="*70)
    
    library = OperatorLibrary(window_sizes=[5, 10, 20])
    
    # Print library
    library.print_library()
    
    # Get operator statistics
    stats = library.get_complexity_stats()
    print("\nOperator Statistics:")
    print(f"  Total: {stats['total_operators']}")
    print(f"  Unary: {stats['unary_operators']}")
    print(f"  Binary: {stats['binary_operators']}")
    print(f"  Avg Complexity: {stats['avg_complexity']:.2f}")
    
    # Access specific operators
    ts_mean = library.get("ts_mean_20")
    print(f"\nOperator 'ts_mean_20':")
    print(f"  Arity: {ts_mean.arity}")
    print(f"  Complexity: {ts_mean.complexity}")
    print(f"  Description: {ts_mean.description}")


def example_simple_expression():
    """Example: Build and evaluate simple expression."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Simple Expression Tree")
    print("="*70)
    
    library = OperatorLibrary()
    builder = ExpressionBuilder(library)
    
    # Build: ts_mean_20(price)
    price_input = builder.add_input("price")
    ts_mean_node = builder.apply_unary("ts_mean_20", price_input)
    
    expr = builder.build(ts_mean_node)
    
    print("\nExpression Tree Built:")
    expr.print_summary()
    
    # Evaluate on synthetic data
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    price_data = pd.Series(100 + np.cumsum(np.random.randn(100) * 2), index=dates)
    
    result = expr.evaluate({"price": price_data})
    
    print(f"\nEvaluation Results:")
    print(f"  Input shape: {price_data.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Output (last 5):\n{result.tail()}")


def example_arithmetic_expression():
    """Example: Arithmetic operations."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Arithmetic Expression")
    print("="*70)
    
    library = OperatorLibrary()
    builder = ExpressionBuilder(library)
    
    # Build: (price - ts_mean_20(price)) / ts_std_20(price)
    # (Price deviation normalized by volatility)
    price_input = builder.add_input("price")
    
    mean_node = builder.apply_unary("ts_mean_20", price_input)
    std_node = builder.apply_unary("ts_std_20", price_input)
    
    deviation = builder.apply_binary("subtract", price_input, mean_node)
    normalized = builder.apply_binary("divide", deviation, std_node)
    
    expr = builder.build(normalized)
    
    print("\nArithmetic Expression Built:")
    expr.print_summary()
    
    # Evaluate
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    price_data = pd.Series(100 + np.cumsum(np.random.randn(100) * 2), index=dates)
    
    result = expr.evaluate({"price": price_data})
    
    print(f"\nOutput Statistics:")
    print(f"  Mean: {result.mean():.4f}")
    print(f"  Std: {result.std():.4f}")
    print(f"  Min: {result.min():.4f}")
    print(f"  Max: {result.max():.4f}")


def example_multi_input_expression():
    """Example: Expression with multiple inputs."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Multi-Input Expression")
    print("="*70)
    
    library = OperatorLibrary()
    builder = ExpressionBuilder(library)
    
    # Build: (price_momentum * volume_momentum)
    # Correlation between price and volume momentum
    price_input = builder.add_input("price")
    volume_input = builder.add_input("volume")
    
    price_mom = builder.apply_unary("ts_momentum_5", price_input)
    volume_mom = builder.apply_unary("ts_momentum_5", volume_input)
    
    combined = builder.apply_binary("multiply", price_mom, volume_mom)
    
    expr = builder.build(combined)
    
    print("\nMulti-Input Expression Built:")
    expr.print_summary()
    
    # Evaluate
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    price_data = pd.Series(100 + np.cumsum(np.random.randn(100) * 2), index=dates)
    volume_data = pd.Series(1000000 + np.cumsum(np.random.randn(100) * 50000), index=dates)
    
    result = expr.evaluate({
        "price": price_data,
        "volume": volume_data,
    })
    
    print(f"\nUsed inputs: {expr.summary()['inputs']}")
    print(f"Output (last 5):\n{result.tail()}")


def example_cross_sectional_expression():
    """Example: Cross-sectional operations."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Cross-Sectional Expression")
    print("="*70)
    
    library = OperatorLibrary()
    builder = ExpressionBuilder(library)
    
    # Build: cs_rank(price)
    # Rank each price cross-sectionally
    price_input = builder.add_input("price")
    ranked = builder.apply_unary("cs_rank", price_input)
    
    expr = builder.build(ranked)
    
    print("\nCross-Sectional Expression Built:")
    expr.print_summary()
    
    # Evaluate on multi-asset data
    dates = pd.date_range(end=datetime.now(), periods=50, freq="D")
    n_assets = 20
    price_data = pd.DataFrame(
        100 + np.cumsum(np.random.randn(50, n_assets) * 2, axis=0),
        index=dates,
        columns=[f"Asset_{i}" for i in range(n_assets)]
    )
    
    result = expr.evaluate({"price": price_data})
    
    print(f"\nOutput Statistics:")
    print(f"  Shape: {result.shape}")
    print(f"  Value range: {result.min().min():.4f} to {result.max().max():.4f}")
    print(f"\nRanked values (last row):\n{result.iloc[-1]}")


def example_complex_expression():
    """Example: Complex nested expression."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Complex Nested Expression")
    print("="*70)
    
    library = OperatorLibrary()
    builder = ExpressionBuilder(library)
    
    # Build: log(abs(ts_momentum_10(price)) + 1)
    price_input = builder.add_input("price")
    
    momentum = builder.apply_unary("ts_momentum_10", price_input)
    abs_momentum = builder.apply_unary("abs", momentum)
    
    # For add with scalar, need to create a workaround
    # Instead: log(abs(ts_momentum_10(price)))
    log_expr = builder.apply_unary("log", abs_momentum)
    
    expr = builder.build(log_expr)
    
    print("\nComplex Expression Built:")
    expr.print_summary()
    
    # Evaluate
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    price_data = pd.Series(100 + np.cumsum(np.random.randn(100) * 2), index=dates)
    
    result = expr.evaluate({"price": price_data})
    
    print(f"\nExpression Formula: {expr.to_string()}")
    print(f"Output (last 5):\n{result.tail()}")


def example_code_generation():
    """Example: Generate Python code from expression."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Code Generation")
    print("="*70)
    
    library = OperatorLibrary()
    builder = ExpressionBuilder(library)
    
    # Build: (price - ts_mean_20(price)) / ts_std_20(price)
    price_input = builder.add_input("price")
    mean_node = builder.apply_unary("ts_mean_20", price_input)
    std_node = builder.apply_unary("ts_std_20", price_input)
    deviation = builder.apply_binary("subtract", price_input, mean_node)
    normalized = builder.apply_binary("divide", deviation, std_node)
    
    expr = builder.build(normalized)
    
    # Generate code
    code = expr.to_python_code(func_name="my_alpha_factor")
    
    print("\nGenerated Python Code:")
    print("-" * 70)
    print(code)
    print("-" * 70)


def example_serialization():
    """Example: JSON serialization."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Serialization")
    print("="*70)
    
    library = OperatorLibrary()
    builder = ExpressionBuilder(library)
    
    # Build expression
    price_input = builder.add_input("price")
    mean_node = builder.apply_unary("ts_mean_20", price_input)
    expr1 = builder.build(mean_node)
    
    # Serialize
    json_str = expr1.to_json()
    print("\nSerialized JSON:")
    print(json_str)
    
    # Deserialize
    expr2 = ExpressionTree.from_json(json_str, library)
    
    print("\nDeserialized Expression:")
    print(f"  Formula: {expr2.to_string()}")
    print(f"  Hash: {expr2.root.hash()}")
    print(f"  Valid: {expr2.is_syntactically_valid()}")


def example_rl_integration():
    """Example: Integration with RL environment."""
    print("\n" + "="*70)
    print("EXAMPLE 9: RL Agent Integration Pattern")
    print("="*70)
    
    library = OperatorLibrary(window_sizes=[5, 10, 20])
    
    print("\nRL Environment Workflow:")
    print("1. RL agent selects action (operator, inputs)")
    print("2. ExpressionBuilder constructs formula incrementally")
    print("3. Evaluate to get factor signal")
    print("4. Calculate reward (IC, Sharpe, Complexity)")
    print("5. Update policy\n")
    
    # Simulate RL agent actions
    print("Simulated Agent Actions:")
    builder = ExpressionBuilder(library)
    
    # Action 1: Add price input
    print("  Step 1: Add 'price' input")
    price_node = builder.add_input("price")
    print(f"    → Node created: {price_node}")
    
    # Action 2: Apply ts_mean_20
    print("  Step 2: Apply 'ts_mean_20' operator")
    mean_node = builder.apply_unary("ts_mean_20", price_node)
    print(f"    → Tree height: {mean_node.height()}, size: {mean_node.size()}")
    
    # Action 3: Apply ts_std_20
    print("  Step 3: Apply 'ts_std_20' operator")
    std_node = builder.apply_unary("ts_std_20", price_node)
    print(f"    → Tree height: {std_node.height()}, size: {std_node.size()}")
    
    # Action 4: Divide (normalize)
    print("  Step 4: Apply 'divide' operator (normalize)")
    final_node = builder.apply_binary("divide", mean_node, std_node)
    print(f"    → Final tree height: {final_node.height()}, size: {final_node.size()}")
    
    # Build and evaluate
    expr = builder.build(final_node)
    
    print(f"\nFinal Formula: {expr.to_string()}")
    print(f"Complexity Score: {expr.root.complexity()}")
    print(f"Validation Status: {expr.get_validation_status()}")


def example_operator_statistics():
    """Example: Operator statistics for RL agent."""
    print("\n" + "="*70)
    print("EXAMPLE 10: Operator Statistics for RL")
    print("="*70)
    
    library = OperatorLibrary()
    
    unary_ops = library.get_unary()
    binary_ops = library.get_binary()
    
    print(f"\nUnary Operators ({len(unary_ops)}):")
    for name, op in list(unary_ops.items())[:5]:
        print(f"  {name:<25} Complexity={op.complexity}")
    print(f"  ... and {len(unary_ops) - 5} more")
    
    print(f"\nBinary Operators ({len(binary_ops)}):")
    for name, op in list(binary_ops.items())[:5]:
        print(f"  {name:<25} Complexity={op.complexity}")
    print(f"  ... and {len(binary_ops) - 5} more")
    
    # Action space for RL
    print(f"\nAction Space for RL Agent:")
    print(f"  Unary operators to choose from: {len(unary_ops)}")
    print(f"  Binary operators to choose from: {len(binary_ops)}")
    print(f"  Total complexity penalty weight: important for limiting expressions")


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█  Operator Module Examples & Integration                 █")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        example_operator_library()
        example_simple_expression()
        example_arithmetic_expression()
        example_multi_input_expression()
        example_cross_sectional_expression()
        example_complex_expression()
        example_code_generation()
        example_serialization()
        example_rl_integration()
        example_operator_statistics()
        
        print("\n✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
