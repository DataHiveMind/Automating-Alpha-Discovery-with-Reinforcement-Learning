"""
Expression tree and DAG generation for formula construction.

Provides:
- Expression tree node representation
- Syntax validation and DAG checking
- Formula parsing and serialization
- Code generation from expressions
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib
import json

import pandas as pd
import numpy as np

from math_ops import Operator, OperatorLibrary


class ExprNode:
    """
    Node in expression DAG.
    
    Represents either:
    - Leaf node: raw data input
    - Operator node: operation applied to children
    """

    _id_counter = 0

    def __init__(
        self,
        operator: Optional[Operator] = None,
        children: Optional[List["ExprNode"]] = None,
        data_input: Optional[str] = None,
    ):
        """
        Initialize expression node.
        
        Args:
            operator: Operator applied at this node
            children: Child nodes (inputs to operator)
            data_input: Name of raw data input (for leaf nodes)
        """
        ExprNode._id_counter += 1
        self.node_id = ExprNode._id_counter
        self.operator = operator
        self.children = children or []
        self.data_input = data_input
        self.memoized_value = None
        self.memoized = False

    def is_leaf(self) -> bool:
        """Check if node is leaf (data input)."""
        return self.data_input is not None

    def is_valid(self) -> bool:
        """Check if node is syntactically valid."""
        if self.is_leaf():
            return self.data_input is not None and len(self.children) == 0
        
        if self.operator is None:
            return False
        
        # Check arity matches
        if len(self.children) != self.operator.arity:
            return False
        
        # Recursively check children
        return all(child.is_valid() for child in self.children)

    def height(self) -> int:
        """Height of subtree."""
        if self.is_leaf():
            return 0
        return 1 + max((child.height() for child in self.children), default=0)

    def size(self) -> int:
        """Number of nodes in subtree."""
        if self.is_leaf():
            return 1
        return 1 + sum(child.size() for child in self.children)

    def complexity(self) -> int:
        """Total complexity of subtree."""
        if self.is_leaf():
            return 0
        op_complexity = self.operator.complexity if self.operator else 0
        child_complexity = sum(child.complexity() for child in self.children)
        return op_complexity + child_complexity

    def get_all_inputs(self) -> Set[str]:
        """Get set of all raw data inputs used."""
        if self.is_leaf():
            return {self.data_input}
        inputs = set()
        for child in self.children:
            inputs.update(child.get_all_inputs())
        return inputs

    def to_string(self) -> str:
        """Convert to human-readable string."""
        if self.is_leaf():
            return self.data_input
        
        if not self.children:
            return self.operator.name
        
        if self.operator.arity == 1:
            return f"{self.operator.name}({self.children[0].to_string()})"
        elif self.operator.arity == 2:
            left = self.children[0].to_string()
            right = self.children[1].to_string()
            
            # Use infix notation for arithmetic operators
            if self.operator.name in ["add", "subtract", "multiply", "divide"]:
                op_symbol = {
                    "add": "+",
                    "subtract": "-",
                    "multiply": "*",
                    "divide": "/",
                }[self.operator.name]
                return f"({left} {op_symbol} {right})"
            else:
                return f"{self.operator.name}({left}, {right})"
        else:
            args = ", ".join(child.to_string() for child in self.children)
            return f"{self.operator.name}({args})"

    def to_json(self) -> Dict[str, Any]:
        """Serialize to JSON."""
        if self.is_leaf():
            return {
                "type": "leaf",
                "data": self.data_input,
            }
        
        return {
            "type": "operator",
            "operator": self.operator.name,
            "children": [child.to_json() for child in self.children],
        }

    @staticmethod
    def from_json(data: Dict[str, Any], op_library: OperatorLibrary) -> "ExprNode":
        """Deserialize from JSON."""
        if data["type"] == "leaf":
            return ExprNode(data_input=data["data"])
        
        operator = op_library.get(data["operator"])
        if operator is None:
            raise ValueError(f"Unknown operator: {data['operator']}")
        
        children = [ExprNode.from_json(child, op_library) for child in data["children"]]
        return ExprNode(operator=operator, children=children)

    def hash(self) -> str:
        """Hash of expression structure."""
        expr_str = self.to_string()
        return hashlib.md5(expr_str.encode()).hexdigest()[:8]

    def __repr__(self):
        return f"ExprNode({self.to_string()}, id={self.node_id})"


class ExpressionTree:
    """
    Complete expression tree with evaluation and validation.
    
    Represents a formula as a DAG that:
    - Takes market data as input
    - Applies operators sequentially
    - Produces alpha factor as output
    """

    def __init__(self, root: ExprNode, op_library: Optional[OperatorLibrary] = None):
        """
        Initialize expression tree.
        
        Args:
            root: Root node of expression
            op_library: Operator library for context
        """
        self.root = root
        self.op_library = op_library or OperatorLibrary()

    def is_valid_dag(self) -> bool:
        """Check if tree forms valid DAG (no cycles)."""
        visited = set()
        
        def has_cycle(node: ExprNode, rec_stack: Set[int]) -> bool:
            visited.add(node.node_id)
            rec_stack.add(node.node_id)
            
            for child in node.children:
                if child.node_id not in visited:
                    if has_cycle(child, rec_stack):
                        return True
                elif child.node_id in rec_stack:
                    return True
            
            rec_stack.remove(node.node_id)
            return False
        
        return not has_cycle(self.root, set())

    def is_syntactically_valid(self) -> bool:
        """Check all nodes are syntactically valid."""
        return self.root.is_valid()

    def get_validation_status(self) -> Dict[str, bool]:
        """Get complete validation status."""
        return {
            "is_valid_dag": self.is_valid_dag(),
            "syntactically_valid": self.is_syntactically_valid(),
            "height_ok": self.root.height() <= 10,
            "size_ok": self.root.size() <= 100,
            "complexity_ok": self.root.complexity() <= 50,
        }

    def evaluate(
        self,
        data: Dict[str, Union[pd.Series, pd.DataFrame]],
        cache_results: bool = True,
    ) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """
        Evaluate expression tree on data.
        
        Args:
            data: Dictionary of {data_name: data_values}
            cache_results: Whether to cache intermediate results
            
        Returns:
            Computed output
        """
        return self._evaluate_node(self.root, data, cache_results)

    def _evaluate_node(
        self,
        node: ExprNode,
        data: Dict[str, Union[pd.Series, pd.DataFrame]],
        cache: bool = True,
    ) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """Recursively evaluate node."""
        # Check cache
        if cache and node.memoized:
            return node.memoized_value
        
        # Leaf node: fetch data
        if node.is_leaf():
            if node.data_input not in data:
                raise ValueError(f"Data '{node.data_input}' not provided")
            result = data[node.data_input]
        
        # Operator node: evaluate children, then operator
        else:
            child_results = [
                self._evaluate_node(child, data, cache)
                for child in node.children
            ]
            result = node.operator(*child_results)
        
        # Cache result
        if cache:
            node.memoized_value = result
            node.memoized = True
        
        return result

    def to_python_code(self, func_name: str = "alpha_factor") -> str:
        """
        Generate Python function code from expression.
        
        Args:
            func_name: Name for generated function
            
        Returns:
            Python function code as string
        """
        inputs = sorted(self.root.get_all_inputs())
        
        code = f"def {func_name}({', '.join(inputs)}):\n"
        code += '    """Auto-generated alpha factor formula."""\n'
        
        # Generate expressions
        expr_dict = {}
        self._generate_code_recursive(self.root, expr_dict, 1)
        
        # Build body
        for var_name, expr_str in expr_dict.items():
            code += f"    {var_name} = {expr_str}\n"
        
        code += f"    return {list(expr_dict.keys())[-1]}\n"
        
        return code

    def _generate_code_recursive(
        self,
        node: ExprNode,
        expr_dict: Dict[str, str],
        counter: List[int],
    ) -> str:
        """Recursively generate code expressions."""
        if node.is_leaf():
            return node.data_input
        
        # Generate for children first
        child_exprs = []
        for child in node.children:
            child_exprs.append(self._generate_code_recursive(child, expr_dict, counter))
        
        # Generate expression for this node
        if node.operator.arity == 1:
            expr = f"{node.operator.name}({child_exprs[0]})"
        elif node.operator.arity == 2:
            if node.operator.name in ["add", "subtract", "multiply", "divide"]:
                op_symbol = {
                    "add": "+",
                    "subtract": "-",
                    "multiply": "*",
                    "divide": "/",
                }[node.operator.name]
                expr = f"({child_exprs[0]} {op_symbol} {child_exprs[1]})"
            else:
                expr = f"{node.operator.name}({child_exprs[0]}, {child_exprs[1]})"
        else:
            expr = f"{node.operator.name}({', '.join(child_exprs)})"
        
        # Store with unique variable name
        var_name = f"_expr_{counter[0]}"
        expr_dict[var_name] = expr
        counter[0] += 1
        
        return var_name

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of expression."""
        return {
            "formula": self.to_string(),
            "height": self.root.height(),
            "size": self.root.size(),
            "complexity": self.root.complexity(),
            "inputs": sorted(self.root.get_all_inputs()),
            "hash": self.root.hash(),
            "validation": self.get_validation_status(),
        }

    def to_string(self) -> str:
        """Convert to string."""
        return self.root.to_string()

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.root.to_json(), indent=2)

    @staticmethod
    def from_json(json_str: str, op_library: Optional[OperatorLibrary] = None) -> "ExpressionTree":
        """Deserialize from JSON string."""
        op_library = op_library or OperatorLibrary()
        data = json.loads(json_str)
        root = ExprNode.from_json(data, op_library)
        return ExpressionTree(root, op_library)

    def print_summary(self):
        """Pretty print summary."""
        summary = self.summary()
        
        print("\n" + "="*70)
        print("EXPRESSION TREE SUMMARY")
        print("="*70)
        print(f"\nFormula: {summary['formula']}")
        print(f"Hash: {summary['hash']}")
        
        print(f"\nStructure:")
        print(f"  Height: {summary['height']}")
        print(f"  Size (nodes): {summary['size']}")
        print(f"  Complexity: {summary['complexity']}")
        
        print(f"\nInputs: {', '.join(summary['inputs'])}")
        
        print(f"\nValidation:")
        for check, passed in summary['validation'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        print("="*70 + "\n")


class ExpressionBuilder:
    """
    Utility for building expressions programmatically.
    
    Used by RL environment to construct expressions step-by-step.
    """

    def __init__(self, op_library: Optional[OperatorLibrary] = None):
        """Initialize builder."""
        self.op_library = op_library or OperatorLibrary()
        self.nodes: List[ExprNode] = []

    def add_input(self, name: str) -> ExprNode:
        """Add raw data input node."""
        node = ExprNode(data_input=name)
        self.nodes.append(node)
        return node

    def apply_unary(self, op_name: str, child: ExprNode) -> ExprNode:
        """Apply unary operator."""
        operator = self.op_library.get(op_name)
        if operator is None:
            raise ValueError(f"Unknown operator: {op_name}")
        if operator.arity != 1:
            raise ValueError(f"Operator {op_name} is not unary")
        
        node = ExprNode(operator=operator, children=[child])
        self.nodes.append(node)
        return node

    def apply_binary(self, op_name: str, left: ExprNode, right: ExprNode) -> ExprNode:
        """Apply binary operator."""
        operator = self.op_library.get(op_name)
        if operator is None:
            raise ValueError(f"Unknown operator: {op_name}")
        if operator.arity != 2:
            raise ValueError(f"Operator {op_name} is not binary")
        
        node = ExprNode(operator=operator, children=[left, right])
        self.nodes.append(node)
        return node

    def build(self, output_node: ExprNode) -> ExpressionTree:
        """Build final expression tree."""
        if not output_node.is_valid():
            raise ValueError("Expression tree is not valid")
        
        return ExpressionTree(output_node, self.op_library)

    def clear(self):
        """Reset builder."""
        self.nodes = []
