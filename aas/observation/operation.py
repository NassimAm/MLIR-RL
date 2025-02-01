from dataclasses import dataclass
from typing import Literal
import os
import subprocess


# ================================================ Oberservable Features ================================================


@dataclass
class NestedLoopFeatures:
    """Dataclass to store the nested loops features data."""
    arg: str
    """The argument representing the loop iterator."""
    lower_bound: int
    """The lower bound of the loop."""
    upper_bound: int
    """The upper bound of the loop."""
    step: int
    """The loop step."""
    iterator_type: Literal["parallel", "reduction"]
    """The type of the loop iterator."""

    def __repr__(self):
        return f"NestedLoopFeatures(arg={self.arg}, lower_bound={self.lower_bound}, upper_bound={self.upper_bound}, step={self.step}, iterator_type={self.iterator_type})"


@dataclass
class OperationFeatures:
    """Dataclass to store the operation features data."""
    operation_type: Literal['matmul', 'conv_2d', 'pooling', 'add', 'generic', 'unknown']
    """The type of the operation (matmul, conv_2d, pooling, add, generic, unknown)."""
    op_count: dict[str, int]
    """Number of arithmetic operations in the operation."""
    op_iter_space_size: int
    """The iteration space size of the operation."""
    nested_loops: list[NestedLoopFeatures]
    """List of nested loops where each loop is represented by the NestedLoopFeatures dataclass."""
    load_data: list[list[str]]
    """List of load accesses where each load is represented by the list of access arguments."""
    store_data: list[list[str]]
    """List of store accesses where each store is represented by the list of access arguments."""

    def __repr__(self):
        return f"OperationFeatures(operation_type={self.operation_type}, op_count={self.op_count}, op_iter_space_size={self.op_iter_space_size}, nested_loops={self.nested_loops}, load_data={self.load_data}, store_data={self.store_data})"


# ================================================ Public functions ================================================


def extract_op_features_from_code(code: str, operation_tag: str):
    """Extracts operation features from the code and operation tag.

    Args:
        code (str): the code to extract features from
        operation_tag (str): the operation tag

    Returns:
        OperationFeatures: extracted operation features
    """
    result = subprocess.run(
        f'{os.getenv("AST_DUMPER_BIN_PATH")} -',
        shell=True,
        input=code.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    raw_ast_info = result.stdout.decode('utf-8')

    return __extract_op_features_from_ast_result(raw_ast_info, operation_tag)


def formula_str_to_list(formula: str):
    """
    Turns assignement formula to a list of (index, factor)
    Example:
        formula = "%x1 - %x2 + %x3 * 5 - %x5 * 3"
        return [('%x1', 1), ('%x2', -1), ('%x3', 5), ('%x5', -3)]

    Args:
        formula (str): the formula as a string input

    Returns:
        list: list of (index, factor) pairs
    """
    formula = formula + ' +'
    terms = formula.split(' ')

    running_factor = 1
    running_term = None

    save = []

    for term in terms:

        if term.startswith('%'):
            running_term = term
        elif term == '+':
            save.append((running_term, running_factor))
            running_factor = 1
        elif term == '-':
            save.append((running_term, running_factor))
            running_factor = -1
        elif term.isnumeric():
            running_factor *= int(term)

    if save[0][0] is None:
        save = save[1:]

    return save


# ================================================ Private functions ================================================

def __extract_op_features_from_ast_result(raw_ast_info: str, operation_tag: str):
    """Extract operation features from the raw ast info.

    Args:
        raw_ast_info (str): the raw ast info
        operation_tag (str): the operation tag

    Returns:
        OperationFeatures: extracted operation features
    """
    info, _ = raw_ast_info.split("########################################")

    operations_lines, _ = info.split('#BEGIN_GRAPH')

    operations_blocks = operations_lines.split('#START_OPERATION')
    operations_blocks = [block.strip() for block in operations_blocks if block]

    operation_block_id = -1
    for i, operation_block_str in enumerate(operations_blocks):
        _, rest = operation_block_str.split("#START_TAG")
        block_operation_tag = rest.strip().split("\n")[0]
        if block_operation_tag == operation_tag:
            operation_block_id = i
            break
    if operation_block_id == -1:
        # raise Exception(f"Operation tag '{operation_tag}' not found in the code.")
        return None

    operation_block = operations_blocks[operation_block_id]
    nested_loops = []
    op_count = {}
    load_data = []
    store_data = []

    raw_operation, rest = operation_block.split("#START_NESTED_LOOPS")

    if 'linalg.matmul' in raw_operation:
        operation_type = 'matmul'
    elif 'linalg.conv' in raw_operation:
        operation_type = 'conv_2d'
    elif 'pooling' in raw_operation:
        operation_type = 'pooling'
    elif 'linalg.add' in raw_operation:
        operation_type = 'add'
    elif 'linalg.generic' in raw_operation:
        operation_type = 'generic'
    else:
        operation_type = 'unknown'

    nested_loops_str, rest = rest.split("#START_LOAD_DATA")
    loop_args = []

    op_iter_space_size = 1
    for nested_loop_str in nested_loops_str.strip().split("\n"):
        if not nested_loop_str:
            continue
        arg, low, high, step, iter = nested_loop_str.strip().split(" ")
        nested_loop = NestedLoopFeatures(
            arg=f'%{arg}',
            lower_bound=int(low),
            upper_bound=int(high),
            step=int(step),
            iterator_type=iter
        )
        nested_loops.append(nested_loop)
        loop_args.append(arg)
        op_iter_space_size *= nested_loop.upper_bound

    loads_data_str, rest = rest.split("#START_OP_COUNT")
    for loop_arg in loop_args:
        loads_data_str = loads_data_str.replace(loop_arg, f'%{loop_arg}')
    for load_data_str in loads_data_str.strip().split("\n"):
        if not load_data_str:
            continue
        load_data.append(load_data_str.split(", "))

    ops_count_str, _ = rest.split("#START_TAG")
    for op_count_str in ops_count_str.strip().split("\n"):
        op, count = op_count_str.strip().split(" ")
        op_count[op] = int(count)

    return OperationFeatures(
        operation_type=operation_type,
        op_count=op_count,
        op_iter_space_size=op_iter_space_size,
        nested_loops=nested_loops,
        load_data=load_data,
        store_data=store_data
    )
